from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tarfile

from .common import COMMIT, ROOT, SOURCE, STATE, binary, environment, read, run, sha, tool, write


def build(args, diagnostic=False):
    env = environment(args.devices, args.rocm)
    for name in ('git', 'cmake', 'ninja', 'hipcc'):
        tool(name, env)
    if not SOURCE.exists():
        SOURCE.mkdir(parents=True)
        run(['git', 'init', SOURCE], env=env, log=STATE / 'git-init.log')
    if not (SOURCE / '.git').is_dir():
        raise RuntimeError('Unidentified MI300X source directory; do not overwrite it.')
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if dirty:
        raise RuntimeError('Private MI300X source has edits; preserve them before rebuilding.')
    available = subprocess.run(['git', '-C', str(SOURCE), 'cat-file', '-e', COMMIT], capture_output=True)
    if available.returncode:
        local = ROOT / 'vendor/llama.cpp'
        cached = local.is_dir() and subprocess.run(
            ['git', '-C', str(local), 'cat-file', '-e', COMMIT], capture_output=True).returncode == 0
        origin = str(local) if cached else 'https://github.com/ggml-org/llama.cpp.git'
        run(['git', '-C', SOURCE, 'fetch', '--depth=1', origin, COMMIT], env=env, log=STATE / 'fetch.log')
    run(['git', '-C', SOURCE, 'checkout', '--detach', COMMIT], env=env, log=STATE / 'checkout.log')
    source = SOURCE
    if diagnostic:
        if not (args.rocm / 'include/rocprofiler-sdk-roctx/roctx.h').is_file():
            raise RuntimeError('Profiling requires ROCprofiler-SDK ROCTx headers and library; inference does not.')
        from .instrumentation import instrument
        source = STATE / 'diagnostic-source'
        source.mkdir(exist_ok=True)
        archive = STATE / 'diagnostic-source.tar'
        with archive.open('wb') as f:
            subprocess.run(['git', '-C', str(SOURCE), 'archive', COMMIT], stdout=f, check=True)
        with tarfile.open(archive) as f:
            f.extractall(source, filter='data')
        archive.unlink()
        (STATE / 'instrumentation.diff').write_text(instrument(source, args.rocm))
    destination = STATE / ('diagnostic-build' if diagnostic else 'build')
    configure = ['cmake', '-S', source, '-B', destination, '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=OFF', '-DGGML_HIP=ON',
                 '-DCMAKE_HIP_ARCHITECTURES=gfx942', f'-DCMAKE_HIP_COMPILER={args.rocm}/llvm/bin/clang++',
                 f'-DCMAKE_PREFIX_PATH={args.rocm}', '-DBUILD_SHARED_LIBS=ON',
                 '-DGGML_BACKEND_DL=OFF', '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                 '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_EXAMPLES=OFF',
                 '-DGGML_HIP_GRAPHS=' + ('OFF' if diagnostic else 'ON')]
    run(configure, env=env, log=destination / 'configure.log')
    run(['cmake', '--build', destination, '--parallel', str(args.jobs), '--target', 'llama-server', 'llama-bench'],
        env=env, log=destination / 'build.log')
    hipcc = tool('hipcc', env)
    helper_flags = [hipcc, '-O2', '-std=c++17', '--offload-arch=gfx942']
    helpers = [STATE / 'device-info']
    if not diagnostic or not helpers[0].exists():
        run([*helper_flags, ROOT / 'benchmark/mi300x/device_info.cpp', '-o', helpers[0]],
            env=env, log=STATE / 'device-info-build.log')
    if diagnostic:
        flags = [f'-I{args.rocm}/include', f'-L{args.rocm}/lib',
                 f'-Wl,-rpath,{args.rocm}/lib', '-lrocprofiler-sdk-roctx']
        for name, output, extra in [('gate', 'libmi300x-gate.so', ['-shared', '-fPIC', '-pthread']),
                                    ('counter_probe', 'counter-probe', [])]:
            helpers.append(STATE / output)
            run([*helper_flags, *extra, ROOT / f'benchmark/mi300x/{name}.cpp', *flags, '-o', STATE / output],
                env=env, log=STATE / f'{name}-build.log')
    artifacts = [p for p in (destination / 'bin').iterdir() if p.is_file() and
                 ('.so' in p.name or p.name in ('llama-server', 'llama-bench'))] + helpers
    write(STATE / ('diagnostic-build.json' if diagnostic else 'build.json'),
          {'backend': 'rocm', 'commit': COMMIT, 'arch': 'gfx942', 'diagnostic': diagnostic,
           'configure': list(map(str, configure)), 'artifacts': {str(p): sha(p) for p in artifacts},
           'instrumentation_sha256': sha(STATE / 'instrumentation.diff') if diagnostic else None})
    print(f'MI300X {"diagnostic" if diagnostic else "inference"} build ready: {binary(diagnostic)}', flush=True)


def manifest(args):
    source = args.manifest or ROOT / f'models/study-manifest-{args.model_size}.json'
    rows = read(source)
    selected = []
    for quant in args.formats:
        matches = [r for r in rows if r['quant'] == quant and r.get('model_size') == args.model_size]
        if len(matches) != 1:
            raise ValueError(f'Manifest must contain exactly one {args.model_size}/{quant} record.')
        row = dict(matches[0])
        if not row.get('sha256') or not row.get('bytes'):
            raise ValueError(f'{quant}: SHA256 and byte size are required.')
        # Only read existing models; downloads go to a distinct AMD model store.
        candidates = [args.model_dir / row['filename']]
        if row.get('path'):
            candidates.append(Path(row['path']))
        candidates.append(ROOT / 'models' / row['filename'])
        row['path'] = str(next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve()))
        selected.append(row)
    return selected


def verify(row):
    path = Path(row['path'])
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise RuntimeError(f'Model size/SHA256 mismatch: {path}')


def prepare(args):
    args.model_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest(args)
    for row in rows:
        path = Path(row['path'])
        if not path.is_file():
            if not row.get('repo') or not row.get('revision'):
                raise RuntimeError(f'{row["quant"]} was supplied locally: copy {row["filename"]} into '
                                   f'{args.model_dir}, or pass --manifest with a pinned source. No substitute model is chosen.')
            from huggingface_hub import hf_hub_download
            parts = row.get('parts', [row])
            downloads = []
            for part in parts:
                cached = Path(hf_hub_download(repo_id=row['repo'], revision=row['revision'],
                                             filename=part['filename']))
                verify(dict(part, path=str(cached)))
                downloads.append(cached)
            temporary = path.with_suffix(path.suffix + '.assembling')
            with temporary.open('wb') as out:
                for cached in downloads:
                    with cached.open('rb') as src:
                        shutil.copyfileobj(src, out, 16 * 1024 * 1024)
            verify(dict(row, path=str(temporary)))
            temporary.replace(path)
        else:
            verify(row)
        write(STATE / f'models/{args.model_size}-{row["quant"]}.json', row)
        print(f'Verified {row["quant"]}: {path}', flush=True)
    return rows


def prepared(args):
    result = []
    for row in manifest(args):
        receipt = STATE / f'models/{args.model_size}-{row["quant"]}.json'
        if not receipt.exists() or read(receipt)['sha256'] != row['sha256']:
            raise RuntimeError(f'Run prepare first for {row["quant"]}.')
        saved = read(receipt)
        verify(saved)  # Verify once per invocation, never on every server restart.
        result.append(saved)
    return result
