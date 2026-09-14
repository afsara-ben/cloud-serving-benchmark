from __future__ import annotations

import csv
import copy
from pathlib import Path
import re
import shutil
import subprocess
import time

from capacity import GIB, estimate

from .common import (STATE, check_build, devices_info, environment, output_root, read, run,
                     protocol_identity, server, sha, tool, wait_file, write)
from .serving import SwapMonitor, burst, placement, server_command
from .metrics import GROUPS, BOTTLE_GROUPS, canonical, definitions, dependencies, extra_counters, names


def csv_rows(path):
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            yield {k.strip().lstrip('\ufeff'): v.strip() if isinstance(v, str) else v
                   for k, v in row.items() if k is not None}


def number(value):
    import math
    result = float(str(value).replace(',', ''))
    if not math.isfinite(result) or result < 0:
        raise ValueError('Non-finite or negative measurement')
    return result


def validate_probe(directory):
    rows = [r for path in directory.rglob('*.csv') for r in csv_rows(path)
            if canonical(r.get('Counter_Name', '')) == 'SQ_WAVES' and 'csb_mi300x_probe' in r.get('Kernel_Name', '')]
    traces = [r for path in directory.rglob('*kernel_trace.csv') for r in csv_rows(path)
              if 'csb_mi300x_probe' in r.get('Kernel_Name', '')]
    return len(rows) == 1 and number(rows[0].get('Counter_Value')) == 64 and len(traces) == 1


def profiler_device_index(directory, pci_bus):
    """Match the profiler's GPU index by PCI address, never by HIP ordinals."""
    matches = set()

    def visit(value):
        if isinstance(value, dict):
            if all(k in value for k in ('logical_node_type_id', 'domain', 'location_id')):
                location = int(value['location_id'])
                address = f'{int(value["domain"]):04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7:x}'
                if address.lower() == pci_bus.lower():
                    matches.add(int(value['logical_node_type_id']))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for path in directory.rglob('*.json'):
        visit(read(path))
    if len(matches) != 1:
        raise RuntimeError(f'Cannot map HIP PCI device {pci_bus} to a unique profiler GPU index in {directory}')
    return str(matches.pop())


def doctor(args):
    check_build(diagnostic=True)
    env = environment(args.devices, args.rocm)
    profiler, avail = tool('rocprofv3', env), tool('rocprofv3-avail', env)
    help_text = subprocess.check_output([profiler, '--help'], env=env, text=True, stderr=subprocess.STDOUT)
    required = ['--selected-regions', '--preload', '--kernel-iteration-range', '--kernel-include-regex',
                '--hip-trace', '--kernel-trace', '--marker-trace', '--memory-copy-trace', '--pmc', '--extra-counters']
    if any(flag not in help_text for flag in required):
        raise RuntimeError('Installed rocprofv3 lacks required capture features; update ROCprofiler-SDK. Inference is independent.')
    directory = STATE / f'doctor/{time.time_ns()}'
    directory.mkdir(parents=True)
    extra = extra_counters(directory / 'extra-counters.yaml')
    write(directory / 'metric-definitions.json', definitions())
    (directory / 'rocprofv3-help.txt').write_text(help_text)
    hardware = devices_info(args)
    group_checks = {}
    for device in args.devices:
        single_env = environment([device], args.rocm)
        out = directory / f'gpu{device}'
        out.mkdir()
        # Proves that loading/warmup kernels are excluded from BOTH traces and
        # counters. Older SDK versions can advertise the flag but fail this test.
        run([profiler, '--selected-regions', '--kernel-trace', '--marker-trace', '--output-format', 'csv', 'json',
             '--output-directory', out / 'probe', '--output-file', 'probe',
             '--kernel-include-regex', 'csb_mi300x_probe', '--kernel-iteration-range', '[1-1]',
             '--extra-counters', extra, '--pmc', 'CSB_SQ_WAVES', '--', STATE / 'counter-probe'],
            env=single_env, log=out / 'probe.log', timeout=120)
        if not validate_probe(out / 'probe'):
            raise RuntimeError(f'ROCm region/counter probe failed on device {device}; inspect {out}. '
                               'Expected exactly one 64-wave dispatch. Profiling stopped before loading a model.')
        if args.action == 'bottlenecks':
            launch_rows = [r for p in (out / 'probe').rglob('*kernel_trace.csv') for r in csv_rows(p)]
            if not launch_rows or any(r.get('Accum_VGPR_Count') in (None, '') for r in launch_rows):
                raise RuntimeError('Bottleneck reports require rocprofv3 VGPR and Accum_VGPR_Count exports; update ROCprofiler-SDK.')
        pci_bus = next(d['pci_bus'] for d in hardware['devices'] if d['physical_hip_index'] == device)
        profiler_index = profiler_device_index(out / 'probe', pci_bus)
        write(out / 'device-binding.json', {'physical_hip_index': device, 'pci_bus': pci_bus,
                                          'profiler_gpu_index': profiler_index})
        run([avail, '-d', profiler_index, 'list', '--pmc'], env=single_env, log=out / 'counter-catalog.txt', timeout=60)
        run([avail, '-d', profiler_index, 'info', '--pmc'], env=single_env, log=out / 'counter-definitions.txt', timeout=60)
        checks = {}
        for name in args.groups:
            command = [avail, '-d', profiler_index, 'pmc-check', *dependencies(name)]
            with (out / f'{name}-check.log').open('w') as f:
                result = subprocess.run(command, env=single_env, stdout=f, stderr=subprocess.STDOUT, timeout=60)
            check = {'available': result.returncode == 0, 'metrics': GROUPS[name],
                     'log': str(out / f'{name}-check.log')}
            # pmc-check examines hardware slot compatibility; this tiny execution
            # also proves that the extra definitions resolve and return scalars.
            if check['available']:
                probe_dir = out / f'{name}-probe'
                try:
                    run([profiler, '--selected-regions', '--kernel-trace', '--output-format', 'csv',
                         '--output-directory', probe_dir, '--kernel-include-regex', 'csb_mi300x_probe',
                         '--kernel-iteration-range', '[1-1]', '--extra-counters', extra,
                         '--pmc', *names(name), '--', STATE / 'counter-probe'],
                        env=single_env, log=out / f'{name}-probe.log', timeout=120)
                    rows = [r for p in probe_dir.rglob('*counter_collection.csv') for r in csv_rows(p)]
                    values = {n: [r for r in rows if canonical(r.get('Counter_Name', '')) == n] for n in GROUPS[name]}
                    check['available'] = all(len(v) == 1 and number(v[0]['Counter_Value']) >= 0 for v in values.values())
                    check['probe'] = str(probe_dir)
                except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
                    check.update(available=False, error=str(error))
            checks[name] = check
        group_checks[device] = checks
    result = {'backend': 'rocm', 'status': 'passed', 'devices': args.devices, 'directory': str(directory),
              'profiler': profiler, 'counter_groups': group_checks, 'extra_counters': str(extra),
              'extra_counters_sha256': sha(extra)}
    write(directory / 'doctor.json', result)
    return result


def selected_cells(args):
    root = output_root(args.output)
    cells = args.cells or sorted((root / 'cells').rglob('cell.json'))
    result = []
    for path in cells:
        path = path / 'cell.json' if path.is_dir() else path
        cell = read(path)
        if cell.get('backend') != 'rocm':
            raise ValueError(f'Expected an AMD serving cell, received {path}')
        cfg = cell['configuration']
        if args.cells or (cfg['quant'] in args.formats and cfg['prompt_tokens'] in args.contexts and
                          cfg['concurrency'] in args.concurrency and cell['repetition'] == 1):
            if cell['status'] == 'completed':
                result.append((path.resolve(), cell))
    if not result:
        raise RuntimeError('No completed MI300X serving cells match the profiling selection.')
    return result


def capture(args, cell_path, cell, kind, directory, doctor_result):
    cfg = cell['configuration']
    if cfg['devices'] != args.devices:
        raise ValueError('Select the same --devices as the serving cell.')
    expected_gpu = cell['gpu_identity']
    before = devices_info(args)
    if [{k: d[k] for k in ('physical_hip_index', 'pci_bus', 'name', 'total_bytes')} for d in before['devices']] != expected_gpu:
        raise ValueError('Serving cell belongs to different GPUs; collect serving measurements on this host first.')
    model = Path(cfg['model_path'])
    if not model.is_file() or model.stat().st_size != cell['model']['bytes']:
        raise ValueError('Serving model missing or size changed.')
    if directory.exists():
        raise RuntimeError(f'Capture already exists: {directory}; use a new --profile-tag.')
    screen = estimate(cell['geometry'], before['devices'], cfg['prompt_tokens'], cfg['concurrency'],
                      cfg['output_tokens'], args.headroom_gib * GIB)
    meta = {'backend': 'rocm', 'status': 'started', 'kind': kind, 'configuration': cfg,
            'source_cell': str(cell_path), 'source_cell_sha256': sha(cell_path),
            'build': check_build(diagnostic=True), 'hardware': before, 'capacity': screen,
            'doctor': doctor_result, 'protocol': protocol_identity(),
            'timing_use': 'Diagnostic; excluded from serving throughput and latency.',
            'capture_phase': getattr(args, 'capture_phase', None), 'capture_batches': args.capture_batches,
            'metric_definitions': definitions()}
    write(directory / 'capture.json', meta)
    try:
        if screen['status'] != 'allocation_validation_required':
            meta['status'] = screen['status']
            return
        gate = directory / 'gate'
        env = environment(args.devices, args.rocm)
        env.update(CSB_MI300X_GATE=str(gate), CSB_MI300X_OPS='1')
        phase = getattr(args, 'capture_phase', None)
        if phase:
            env.update(CSB_MI300X_PHASE=phase, CSB_MI300X_BATCHES=str(args.capture_batches),
                       CSB_MI300X_WIDTH=str(cfg['concurrency'] if phase == 'decode' else 0))
        command = [doctor_result['profiler'], '--preload', str(STATE / 'libmi300x-gate.so'),
                   '--selected-regions', '--hip-trace', '--kernel-trace', '--marker-trace', '--memory-copy-trace',
                   '--output-format', 'csv', '--output-directory', str(directory / 'rocprof'), '--output-file', 'capture']
        if kind != 'trace':
            command += ['--kernel-include-regex', args.kernel_regex, '--extra-counters', doctor_result['extra_counters']]
            if not getattr(args, 'counter_uncapped', False):
                command += ['--kernel-iteration-range', f'[{args.launch_skip + 1}-{args.launch_skip + args.launch_count}]']
            command += ['--pmc', *names(kind)]
        command += ['--', *server_command(cfg, args.port, diagnostic=True)]
        meta['command'] = command
        prompts_path = cell_path.parent / 'prompts.json'
        prompts = read(prompts_path)[:cfg['concurrency']]
        if len(prompts) != cfg['concurrency']:
            raise RuntimeError('Saved serving prompt fixture is incomplete.')
        import hashlib
        import json
        for p in prompts:
            digest = hashlib.sha256(json.dumps(p['messages'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            if p['prompt_tokens'] != cfg['prompt_tokens'] or digest != p['input_sha256']:
                raise RuntimeError('Saved prompt fixture count/hash mismatch.')
        meta['prompts_sha256'] = sha(prompts_path)
        base_url = f'http://127.0.0.1:{args.port}'
        with server(command, env, directory, args.port, args.startup_timeout) as process:
            pid = int(wait_file(gate.with_suffix('.ready'), process))
            meta['placement'] = placement((directory / 'server.log').read_text(), cell['geometry'], cfg)
            if not meta['placement']['valid']:
                raise RuntimeError('; '.join(meta['placement']['errors']))
            warmup = burst(cfg, prompts, base_url, args.timeout, warmup=True)
            write(directory / 'warmup.json', warmup)
            if not warmup['valid']:
                raise RuntimeError('Profile warmup validation failed.')
            gate.with_suffix('.start').touch()
            wait_file(gate.with_suffix('.started'), process)
            with SwapMonitor(pid) as monitor:
                result = burst(cfg, prompts, base_url, args.timeout)
            gate.with_suffix('.stop').touch()
            wait_file(gate.with_suffix('.stopped'), process)
            if phase:
                count = int(gate.with_suffix('.batches').read_text()) if gate.with_suffix('.batches').exists() else 0
                meta['captured_batches'] = count
                if count != args.capture_batches:
                    raise RuntimeError(f'Expected {args.capture_batches} pure {phase} batches, captured {count}.')
            write(directory / 'requests.json', result)
            if not result['valid'] or monitor.peak or monitor.error:
                raise RuntimeError('Profile request or host-swap validation failed.')
        # Context manager closes the server; rocprofv3 must flush successfully.
        meta['profiler_exit_code'] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f'Profiler/server exit {process.returncode}; capture may be truncated.')
        from .report import normalize_capture
        normalized = normalize_capture(directory)
        write(directory / 'kernels.json', normalized)
        meta['kernels_sha256'] = sha(directory / 'kernels.json')
        meta['raw_files'] = {str(p.relative_to(directory)): sha(p) for p in sorted((directory / 'rocprof').rglob('*.csv'))}
        if not normalized['kernels']:
            raise RuntimeError('No matching kernel dispatches were exported.')
        if kind != 'trace' and not any(k['metrics'] for k in normalized['kernels']):
            raise RuntimeError('Counter capture contains no hardware-counter rows.')
        if not any(k['phase'] in ('prefill', 'decode', 'mixed') and k['role'] != 'unattributed' for k in normalized['kernels']):
            raise RuntimeError('No kernels could be attributed through HIP launch correlations and ROCTx ranges.')
        meta['status'] = 'captured'
    except KeyboardInterrupt:
        meta['status'] = 'interrupted'
        raise
    except Exception as error:
        meta.update(status='failed', error=str(error))
    finally:
        write(directory / 'capture.json', meta)


def profile(args):
    cells = selected_cells(args)
    # Hash once per model, not once per counter pass.
    for path, digest in {(c['configuration']['model_path'], c['configuration']['model_sha256']) for _, c in cells}:
        if sha(path) != digest:
            raise RuntimeError(f'Serving model SHA256 changed: {path}')
    result = doctor(args)
    root = output_root(args.output)
    failures = []
    for path, cell in cells:
        cfg = cell['configuration']
        for kind in ['trace', *args.groups]:
            directory = root / f'profiles/{args.profile_tag}/{cfg["model_size"]}/{cfg["quant"]}/p{cfg["prompt_tokens"]}/c{cfg["concurrency"]}/r{cell["repetition"]}/{kind}'
            if kind != 'trace' and not all(result['counter_groups'][d][kind]['available'] for d in args.devices):
                if directory.exists():
                    raise RuntimeError(f'Capture exists: {directory}; use a new --profile-tag.')
                write(directory / 'capture.json', {'backend': 'rocm', 'status': 'unsupported_counter_group',
                                                  'kind': kind, 'configuration': cfg, 'doctor': result})
                failures.append(str(directory))
                continue
            print(f'Profiling {cfg["quant"]} P={cfg["prompt_tokens"]} C={cfg["concurrency"]}: {kind}', flush=True)
            capture(args, path, cell, kind, directory, result)
            if read(directory / 'capture.json')['status'] != 'captured':
                failures.append(str(directory))
    return failures


def bottlenecks(args):
    """Pure prefill and full-concurrency decode batches, all GPUs and operators.

    No per-kernel launch cap: selecting the first few names can exclude GPU1,
    decode or late-layer operators. The report instead selects matched geometry.
    """
    cells = selected_cells(args)
    for path, digest in {(c['configuration']['model_path'], c['configuration']['model_sha256']) for _, c in cells}:
        if sha(path) != digest:
            raise RuntimeError(f'Serving model SHA256 changed: {path}')
    options = copy.copy(args)
    options.groups = list(BOTTLE_GROUPS)
    options.counter_uncapped = True
    options.kernel_regex = '.*'
    result = doctor(options)
    root = output_root(args.output)
    # Retain the actual installed catalog and scalar probes with the results,
    # so copying the study does not lose the definitions of native HBM counters.
    saved_doctor = root / 'profiles' / args.profile_tag / 'profiler-info'
    shutil.copytree(result['directory'], saved_doctor)
    result = dict(result, extra_counters=str(saved_doctor / 'extra-counters.yaml'),
                  saved_evidence=str(saved_doctor))
    failures = []
    for path, cell in cells:
        cfg = cell['configuration']
        for phase in ('prefill', 'decode'):
            options.capture_phase = phase
            for kind in ('trace', *options.groups):
                directory = root / f'profiles/{args.profile_tag}/{cfg["model_size"]}/{cfg["quant"]}/p{cfg["prompt_tokens"]}/c{cfg["concurrency"]}/r{cell["repetition"]}/{phase}/{kind}'
                if directory.exists():
                    raise RuntimeError(f'Capture exists: {directory}; use a new --profile-tag.')
                if kind != 'trace' and not all(result['counter_groups'][d][kind]['available'] for d in args.devices):
                    write(directory / 'capture.json', {'backend': 'rocm', 'status': 'unsupported_counter_group',
                          'kind': kind, 'configuration': cfg, 'capture_phase': phase, 'doctor': result})
                    failures.append(str(directory))
                    continue
                print(f'Bottlenecks {cfg["quant"]} P={cfg["prompt_tokens"]} C={cfg["concurrency"]} {phase}: {kind}', flush=True)
                capture(options, path, cell, kind, directory, result)
                if read(directory / 'capture.json')['status'] != 'captured':
                    failures.append(str(directory))
    return failures


def compute(args):
    """Optional vendor full-counter/roofline analysis of a bounded llama-bench.

    This evaluates one HIP GPU, synthetic tokens and all selected dispatches,
    including benchmark initialization. It is not the HTTP serving workload.
    """
    from .setup import prepared
    check_build()
    if len(args.devices) != 1:
        raise ValueError('rocprof-compute microbenchmarks use one --devices index per invocation.')
    info = devices_info(args)
    env = environment(args.devices, args.rocm)
    profiler = tool('rocprof-compute', env)
    root = output_root(args.output)
    directory = root / 'compute' / args.profile_tag
    directory.mkdir(parents=True, exist_ok=False)
    help_text = subprocess.check_output([profiler, 'profile', '--help'], env=env, text=True, stderr=subprocess.STDOUT)
    if '--output-directory' not in help_text:
        raise RuntimeError('This rocprof-compute release lacks --output-directory; update the profiler.')
    for row in prepared(args):
        for p in args.contexts:
            destination = directory / f'{row["quant"]}-p{p}'
            if p + args.output_tokens >= row.get('native_context', 131072):
                write(destination / 'status.json', {'backend': 'rocm', 'status': 'native_context_unsupported'})
                continue
            # One synthetic sequence, FP16 KV; working buffers validated by the benchmark.
            minimum = row['bytes'] + slot_bytes(row, p + args.output_tokens) + args.headroom_gib * GIB
            if minimum > info['devices'][0]['free_bytes']:
                write(destination / 'status.json', {'backend': 'rocm', 'status': 'capacity_estimated'})
                continue
            bench = STATE / 'build/bin/llama-bench'
            command = [profiler, 'profile', '-n', f'mi300x-{row["quant"]}-p{p}', '--output-directory', str(destination),
                       '-k', args.kernel_regex_compute, '--', str(bench), '-m', row['path'], '-p', '0', '-n', '0',
                       '-pg', f'{p},{args.output_tokens}', '-ngl', '99', '-fa', 'on', '-r', '1', '-lm', 'none',
                       '-ctk', 'f16', '-ctv', 'f16', '-b', str(args.batch), '-ub', str(args.ubatch)]
            write(directory / f'{row["quant"]}-p{p}-command.json', {'backend': 'rocm', 'command': command,
                  'scope': 'Synthetic single-sequence llama-bench; includes initialization; not serving performance.'})
            run(command, env=env, log=directory / f'{row["quant"]}-p{p}-profile.log', timeout=args.compute_timeout)
            data_dirs = sorted({f.parent for f in destination.rglob('sysinfo.csv')})
            if not data_dirs:
                raise RuntimeError(f'rocprof-compute did not export sysinfo.csv under {destination}.')
            for i, data in enumerate(data_dirs):
                run([profiler, 'analyze', '--path', str(data)], env=env,
                    log=directory / f'{row["quant"]}-p{p}-analysis-{i}.txt', timeout=args.compute_timeout)


def slot_bytes(row, tokens):
    return tokens * row['layers'] * row['kv_heads'] * row['head_dimension'] * 4
