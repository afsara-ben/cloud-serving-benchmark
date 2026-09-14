from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import threading
import time

# Only backend-neutral GGUF arithmetic and HTTP workload code are reused.
from capacity import GIB, estimate, inspect_model, layer_devices, process_swap_bytes, slot_capacity
import load_test

from .common import (ROOT, STATE, binary, check_build, devices_info, environment, output_root,
                     protocol_identity, read, server, write)
from .setup import manifest, prepared


def configuration(args, row, prompt, concurrency):
    return {'backend': 'rocm', 'model_size': args.model_size, 'quant': row['quant'],
            'model_path': row['path'], 'model_sha256': row['sha256'], 'prompt_tokens': prompt,
            'output_tokens': args.output_tokens, 'concurrency': concurrency, 'requests': 2 * concurrency,
            'devices': args.devices, 'split': args.split, 'batch': args.batch, 'ubatch': args.ubatch,
            'kv_type': 'f16', 'seed': args.seed, 'chat_template_file': row.get('chat_template_file')}


def server_command(cfg, port, diagnostic=False):
    command = [str(binary(diagnostic)), '--model', cfg['model_path'], '--alias', 'mi300x-benchmark',
               '--host', '127.0.0.1', '--port', str(port), '--n-gpu-layers', '99',
               '--parallel', str(cfg['concurrency']), '--ctx-size',
               str(cfg['concurrency'] * slot_capacity(cfg['prompt_tokens'], cfg['output_tokens'])),
               '--batch-size', str(cfg['batch']), '--ubatch-size', str(cfg['ubatch']),
               '--flash-attn', 'on', '--cont-batching', '--no-cache-prompt', '--cache-ram', '0',
               '--no-cache-idle-slots', '--metrics', '--jinja', '--split-mode', 'layer',
               '--tensor-split', ','.join(map(str, cfg['split'])), '--main-gpu', '0',
               '--load-mode', 'none', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
               '--no-context-shift', '--fit', 'off', '--log-verbosity', '4']
    if cfg.get('chat_template_file'):
        command += ['--chat-template-file', str(ROOT / cfg['chat_template_file'])]
    return command


def placement(log, metadata, cfg):
    errors = []
    offload = re.search(r'offloaded (\d+)/(\d+) layers to GPU', log)
    layers = metadata['layers'] + 1
    if not offload or tuple(map(int, offload.groups())) != (layers, layers):
        errors.append('Full transformer/output layer GPU offload not established.')
    for regex, expected in [(r'n_seq_max\s*=\s*(\d+)', cfg['concurrency']),
                            (r'n_ctx_(?:per_seq|seq)\s*=\s*(\d+)', slot_capacity(cfg['prompt_tokens'], cfg['output_tokens']))]:
        match = re.search(regex, log)
        if not match or int(match[1]) != expected:
            errors.append(f'Incorrect/missing slot configuration: {regex}')
    buffers = re.findall(r'(ROCm\d+|CPU)\s+KV buffer size\s*=\s*([\d.]+) MiB', log)
    expected = set(metadata['layer_devices'][:-1])
    if {int(name[4:]) for name, _ in buffers if name.startswith('ROCm')} != expected:
        errors.append('Missing HIP KV allocation evidence for the selected layer split.')
    if any(name == 'CPU' and float(size) > 0 for name, size in buffers):
        errors.append('CPU KV fallback detected.')
    types = re.search(r'K \(([^)]+)\):[^\n]+V \(([^)]+)\):', log)
    if not types or types.groups() != ('f16', 'f16'):
        errors.append('Positive FP16 K/V allocation evidence missing.')
    if re.search(r'CUDA\d+\s+(?:model|KV) buffer', log):
        errors.append('CUDA allocation in a HIP-only run.')
    return {'valid': not errors, 'errors': errors, 'kv_buffers_mib': dict(buffers)}


def make_prompts(cfg, base_url, timeout):
    topics = [s.strip() for s in (ROOT / 'prompts/topics.txt').read_text().splitlines() if s.strip()]
    tokenizer = load_test.ServerTokenizer(base_url, timeout)
    prompts = []
    for i in range(cfg['requests']):
        messages, count = load_test.make_messages(tokenizer, topics[i % len(topics)], i,
                                                 cfg['prompt_tokens'], exact=True)
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False).encode()).hexdigest()
        prompts.append({'messages': messages, 'prompt_tokens': count, 'input_sha256': digest})
    return prompts


def burst(cfg, prompts, base_url, timeout, *, warmup=False):
    count = cfg['concurrency'] if warmup else len(prompts)
    tokens = min(16, cfg['output_tokens']) if warmup else cfg['output_tokens']
    barrier = threading.Barrier(cfg['concurrency'])

    def one(i):
        if i < cfg['concurrency']:
            barrier.wait(timeout=timeout)
        p = prompts[i]
        r = load_test.stream_completion(base_url, 'mi300x-benchmark', p['messages'], tokens, timeout,
                                        cfg['seed'] + i + (100000 if warmup else 0), cache_prompt=False)
        load_test.finalize_record(r, p['prompt_tokens'], tokens, cache_policy='forbid',
                                 expected_prompt_tokens=cfg['prompt_tokens'])
        r.update(request_index=i, input_sha256=p['input_sha256'])
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
        records = list(pool.map(one, range(count)))
    start = min(r['_started'] for r in records)
    wall = max(r['_finished'] for r in records) - start
    started_unix = min(r['started_unix_seconds'] for r in records)
    for r in records:
        r['start_offset_seconds'] = r.pop('_started') - start
        r['end_offset_seconds'] = r.pop('_finished') - start
    good = [r for r in records if r['ok']]
    summary = {'successful_requests': len(good), 'failed_requests': count - len(good),
               'wall_seconds': wall, 'started_unix_seconds': started_unix,
               'finished_unix_seconds': started_unix + wall,
               'output_tokens_per_second': sum(r['completion_tokens'] for r in good) / wall,
               **load_test.interval_statistics(records)}
    for field, key in [('ttft_seconds', 'ttft'), ('tpot_seconds', 'tpot'), ('duration_seconds', 'e2e')]:
        values = [r[field] for r in good if r.get(field) is not None]
        for quantile, suffix in [(.5, 'p50'), (.95, 'p95')]:
            summary[f'{key}_{suffix}_ms'] = load_test.percentile(values, quantile) * 1000 if values else None
    valid = len(good) == count and summary.get('max_client_inflight') == cfg['concurrency']
    return {'backend': 'rocm', 'valid': valid, 'configuration': cfg, 'warmup': warmup,
            'summary': summary, 'requests': records}


def grid(args):
    # No hardware calls, build tools or GPU dependencies: usable on a CUDA host.
    jobs = []
    for row in manifest(args):
        placement_map = layer_devices(row['layers'], args.split)
        for p in args.contexts:
            for c in args.concurrency:
                kv_per_layer = c * slot_capacity(p, args.output_tokens) * row['kv_heads'] * row['head_dimension'] * 4
                jobs.append({'quant': row['quant'], 'prompt_tokens': p, 'concurrency': c,
                             'output_tokens': args.output_tokens, 'repetitions': args.repetitions,
                             'weight_file_gib': row['bytes'] / GIB,
                             'kv_gib_per_gpu': [placement_map[:-1].count(i) * kv_per_layer / GIB
                                                for i in range(len(args.devices))],
                             'allocation': 'Requires MI300X free-memory check and actual server allocation.'})
    return {'backend': 'rocm', 'devices': args.devices, 'split': args.split, 'jobs': jobs,
            'cell_count': len(jobs) * args.repetitions, 'profiling_not_in_serving_timings': True}


class SwapMonitor:
    def __init__(self, pid):
        self.pid, self.peak, self.error = pid, 0, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, process_swap_bytes(self.pid))
            except (OSError, RuntimeError) as error:
                self.error = str(error)
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def serve(args):
    root = output_root(args.output)
    build = check_build()
    rows = prepared(args)
    hardware = devices_info(args)
    study_identity = {'model_size': args.model_size, 'devices': args.devices, 'split': args.split,
                      'output_tokens': args.output_tokens, 'batch': args.batch, 'ubatch': args.ubatch,
                      'seed': args.seed, 'build': build, 'protocol': protocol_identity(),
                      'gpu_identity': [{k: d[k] for k in ('physical_hip_index', 'pci_bus', 'name', 'total_bytes')}
                                       for d in hardware['devices']]}
    marker = read(root / 'mi300x-study.json')
    if marker.get('serving_identity') is not None and marker['serving_identity'] != study_identity:
        raise RuntimeError('Output belongs to a different serving build/hardware/protocol; select a new --output.')
    write(root / 'mi300x-study.json', dict(marker, serving_identity=study_identity))
    write(root / 'plan.json', grid(args))
    failures = []
    for row in rows:
        metadata = inspect_model(row['path'], row, STATE / 'source', args.split)
        for p in args.contexts:
            for c in args.concurrency:
                cfg = configuration(args, row, p, c)
                for repetition in range(1, args.repetitions + 1):
                    directory = root / f'cells/{args.model_size}/{row["quant"]}/p{p}/c{c}/r{repetition}'
                    cell_file = directory / 'cell.json'
                    identity = {'configuration': cfg, 'build': build, 'protocol': study_identity['protocol'], 'repetition': repetition,
                                'gpu_identity': [{k: d[k] for k in ('physical_hip_index', 'pci_bus', 'name', 'total_bytes')}
                                                 for d in hardware['devices']]}
                    if directory.exists():
                        previous = read(cell_file) if cell_file.exists() else {}
                        if args.resume and previous.get('status') == 'completed' and all(previous.get(k) == v for k, v in identity.items()):
                            print(f'Resuming: keep {directory}', flush=True)
                            continue
                        raise RuntimeError(f'Cell already exists or differs: {directory}; choose a new output or --resume for completed identical cells.')
                    cell = {'backend': 'rocm', **identity, 'model': row, 'geometry': metadata, 'status': 'started'}
                    write(cell_file, cell)
                    print(f'MI300X {row["quant"]}: P={p} C={c} repetition={repetition}', flush=True)
                    try:
                        before = devices_info(args)
                        cell['hardware_before'] = before
                        screen = estimate(metadata, before['devices'], p, c, args.output_tokens, args.headroom_gib * GIB)
                        cell['capacity'] = screen
                        if screen['status'] != 'allocation_validation_required':
                            cell['status'] = screen['status']
                            write(cell_file, cell)
                            continue
                        command = server_command(cfg, args.port)
                        cell['command'] = command
                        with server(command, environment(args.devices, args.rocm), directory, args.port, args.startup_timeout) as process:
                            cell['placement'] = placement((directory / 'server.log').read_text(), metadata, cfg)
                            if not cell['placement']['valid']:
                                raise RuntimeError('; '.join(cell['placement']['errors']))
                            cell['hardware_loaded'] = devices_info(args)
                            base_url = f'http://127.0.0.1:{args.port}'
                            prompts = make_prompts(cfg, base_url, args.timeout)
                            write(directory / 'prompts.json', prompts)
                            warmup = burst(cfg, prompts, base_url, args.timeout, warmup=True)
                            write(directory / 'warmup.json', warmup)
                            if not warmup['valid']:
                                raise RuntimeError('Warmup failed exact-token, cache or concurrency validation.')
                            with SwapMonitor(process.pid) as monitor:
                                result = burst(cfg, prompts, base_url, args.timeout)
                            write(directory / 'requests.json', result)
                            cell['peak_process_swap_bytes'] = monitor.peak
                            if monitor.error or monitor.peak:
                                raise RuntimeError(f'Host swap evidence invalid: {monitor.error or monitor.peak}')
                            if not result['valid']:
                                raise RuntimeError('Measured requests failed exact-token, cache or concurrency validation.')
                            cell['summary'] = result['summary']
                            cell['status'] = 'completed'
                    except KeyboardInterrupt:
                        cell['status'] = 'interrupted'
                        raise
                    except Exception as error:
                        log = (directory / 'server.log').read_text(errors='replace') if (directory / 'server.log').exists() else ''
                        cell['status'] = 'observed_oom' if re.search(r'out of memory|failed to allocate|hipMalloc.*fail', log, re.I) else 'failed'
                        cell['error'] = str(error)
                        failures.append(str(directory))
                    finally:
                        write(cell_file, cell)
    return failures
