from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.run/mi300x'
SOURCE = STATE / 'source'
COMMIT = '3f5e94d7c2ab2267fe39852051777fe30c1f49ef'
FORMATS = ('IQ1_M', 'Q2_K', 'Q4_K_M', 'Q8_0')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protocol_identity():
    files = [*sorted((ROOT / 'benchmark/mi300x').glob('*.py')),
             ROOT / 'benchmark/load_test.py', ROOT / 'benchmark/capacity.py', ROOT / 'scripts/run_mi300x.py']
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
            'python': platform.python_version(), 'system': platform.platform(), 'cpu_count': os.cpu_count()}


def environment(devices, rocm):
    # Child-local isolation: do not inherit CUDA selection, build flags, preloads,
    # tuning switches or active profiler sessions. Never change the parent shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CUDA', 'NVIDIA', 'NSYS', 'NCU_', 'ROCP', 'ROCM_', 'HSA_', 'GGML_', 'CSB_', 'HIP_', 'ROCR_', 'LLAMA_ARG_'))
        and k not in {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'GPU_DEVICE_ORDINAL', 'ACCELERATOR_BACKEND',
                     'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'HIPCXX', 'CUDACXX',
                     'CMAKE_PREFIX_PATH', 'CMAKE_ARGS'}}
    env.update(HIP_VISIBLE_DEVICES=','.join(devices), HIP_PLATFORM='amd', ROCM_PATH=str(rocm),
               HIP_PATH=str(rocm), HIPCXX=str(rocm / 'llvm/bin/clang++'),
               PATH=f'{rocm}/bin:{rocm}/llvm/bin:' + env.get('PATH', ''),
               LD_LIBRARY_PATH=f'{rocm}/lib:{rocm}/lib64')
    return env


def run(command, *, env, log=None, cwd=None, timeout=None):
    command = list(map(str, command))
    print('Running ' + command[0] + (f' (log: {log})' if log else ''), flush=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open('w') as f:
            result = subprocess.run(command, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Exit {result.returncode}: see {log}')
        return result
    return subprocess.run(command, env=env, cwd=cwd, check=True, timeout=timeout)


def tool(name, env):
    path = shutil.which(name, path=env['PATH'])
    if not path:
        raise RuntimeError(f'MI300X stage needs {name}; install it in the ROCm environment.')
    return path


def binary(diagnostic=False):
    return STATE / ('diagnostic-build' if diagnostic else 'build') / 'bin/llama-server'


def check_build(diagnostic=False):
    meta = read(STATE / ('diagnostic-build.json' if diagnostic else 'build.json'))
    if meta.get('backend') != 'rocm' or meta.get('commit') != COMMIT:
        raise RuntimeError('Incorrect MI300X build provenance; run setup again.')
    for path, digest in meta['artifacts'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Build artifact changed: {path}; run setup again.')
    return meta


def devices_info(args):
    result = subprocess.run([str(STATE / 'device-info')], env=environment(args.devices, args.rocm),
                            capture_output=True, text=True, timeout=60, check=True)
    info = json.loads(result.stdout)
    if len(info['devices']) != len(args.devices):
        raise RuntimeError('HIP visibility did not expose exactly the requested devices.')
    for physical, d in zip(args.devices, info['devices']):
        d['physical_hip_index'] = physical
        if d['arch'].split(':')[0] != 'gfx942' or 'MI300X' not in d['name'].upper():
            raise RuntimeError(f'Expected MI300X/gfx942; found {d["name"]}/{d["arch"]}.')
        if d['compute_units'] != 304 or d['total_bytes'] < 180 * 1024**3:
            raise RuntimeError('Expected a full MI300X device (304 CUs, approximately 192 GiB). '
                               'Partitioned devices need a separate capacity/roof configuration.')
    return info


@contextlib.contextmanager
def lock():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'pipeline.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another MI300X pipeline is running in this checkout.') from None
        yield


def output_root(path):
    path = Path(path).resolve()
    marker = path / 'mi300x-study.json'
    if marker.exists():
        if read(marker).get('backend') != 'rocm':
            raise RuntimeError('Output directory belongs to another backend.')
    elif path.exists() and any(path.iterdir()):
        raise RuntimeError('Refusing an existing unmarked output directory; choose a new MI300X output.')
    else:
        write(marker, {'backend': 'rocm', 'hardware': 'MI300X', 'schema_version': 1, 'commit': COMMIT})
    return path


def ensure_port(port):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', port))
        except OSError:
            raise RuntimeError(f'Port {port} is occupied; choose another --port. No process was stopped.') from None


def wait_file(path, process, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        error = path.with_suffix('.error')
        if error.exists():
            raise RuntimeError(error.read_text())
        if path.exists():
            return path.read_text()
        if process.poll() is not None:
            raise RuntimeError('Server/profiler exited while waiting for gate acknowledgement.')
        time.sleep(.1)
    raise RuntimeError(f'Timed out waiting for {path.name}')


@contextlib.contextmanager
def server(command, env, directory, port, timeout):
    ensure_port(port)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'server.log').open('w') as log:
        process = subprocess.Popen(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited during startup; see {directory / "server.log"}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError('Server startup timed out.')
            yield process
        finally:
            # Only our process group, never a generic llama-server PID file or pkill.
            if process.poll() is None:
                # Signal the target first so the profiler parent can flush CSVs.
                target_pid = process.pid
                gate = env.get('CSB_MI300X_GATE')
                if gate and Path(gate + '.ready').exists():
                    candidate = int(Path(gate + '.ready').read_text())
                    try:
                        if os.getpgid(candidate) == process.pid:
                            target_pid = candidate
                    except ProcessLookupError:
                        pass
                try:
                    os.kill(target_pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
