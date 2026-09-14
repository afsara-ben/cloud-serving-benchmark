"""CPU-only validation and aggregation for the AMD publication exports."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import load_test
from .common import read, sha
from .metrics import BOTTLE_GROUPS
from .report import coordinates


def delta(reference, value):
    return 100 * (value / reference - 1) if reference and value is not None else None


def finite(value):
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise ValueError('Expected a finite nonnegative metric')
    return n


def validate_serving(path):
    """Recompute plot values from the retained request records, not summary flags."""
    cell = read(path)
    if cell.get('backend') != 'rocm' or cell.get('status') != 'completed':
        raise ValueError('Expected a completed AMD serving cell')
    cfg = cell['configuration']
    requests = read(path.parent / 'requests.json')
    prompts = read(path.parent / 'prompts.json')
    warmup = read(path.parent / 'warmup.json')
    if not cell.get('placement', {}).get('valid') or cell.get('peak_process_swap_bytes') != 0:
        raise ValueError('Missing full GPU placement / zero-swap evidence')
    expected = 2 * cfg['concurrency']
    if len(prompts) != expected or len(requests['requests']) != expected:
        raise ValueError('Expected exactly 2C measured prompts and requests')
    hashes = []
    for p in prompts:
        digest = hashlib.sha256(json.dumps(p['messages'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        if digest != p['input_sha256'] or p['prompt_tokens'] != cfg['prompt_tokens']:
            raise ValueError('Prompt fixture hash/count mismatch')
        hashes.append(digest)
    for burst, count, outputs in [(requests, expected, cfg['output_tokens']),
                                   (warmup, cfg['concurrency'], min(16, cfg['output_tokens']))]:
        records = burst['requests']
        if len(records) != count or not burst.get('valid'):
            raise ValueError('Invalid saved request burst')
        for i, r in enumerate(records):
            checked = dict(r)
            load_test.finalize_record(checked, prompts[i]['prompt_tokens'], outputs,
                                      cache_policy='forbid', expected_prompt_tokens=cfg['prompt_tokens'])
            if (not r.get('ok') or r.get('completion_tokens') != outputs or
                    r.get('prompt_tokens') != cfg['prompt_tokens'] or
                    r.get('evaluated_prompt_tokens') != cfg['prompt_tokens'] or
                    r.get('cached_prompt_tokens') != 0 or r.get('input_sha256') != hashes[i]):
                raise ValueError('Request has wrong tokens, cached inputs, errors or changed prompt')
            if (not checked['ok'] or not r.get('stream_done_received') or not r.get('stream_content_events') or
                    not math.isclose(r['duration_seconds'], r['end_offset_seconds'] - r['start_offset_seconds'], rel_tol=1e-8, abs_tol=1e-7) or
                    not math.isclose(r['tpot_seconds'], checked['tpot_seconds'], rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError('Stream, server usage or derived timing validation failed')
        if load_test.interval_statistics(records)['max_client_inflight'] != cfg['concurrency']:
            raise ValueError('Measured concurrency differs from requested concurrency')
    records = requests['requests']
    wall = max(r['end_offset_seconds'] for r in records) - min(r['start_offset_seconds'] for r in records)
    if wall <= 0:
        raise ValueError('Invalid request wall time')
    values = {'output_tokens_per_second': expected * cfg['output_tokens'] / wall}
    for name in ('ttft', 'tpot'):
        values[name + '_p95_ms'] = 1000 * load_test.percentile([finite(r[name + '_seconds']) for r in records], .95)
    for name, value in values.items():
        if not math.isclose(value, cell['summary'][name], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f'Saved summary disagrees with requests: {name}')
    return {**values, 'prompt_hashes': hashes, 'source': str(path), 'cell_sha256': sha(path),
            'request_sha256': sha(path.parent / 'requests.json'), 'configuration': cfg,
            'repetition': cell['repetition'], 'gpu_identity': cell['gpu_identity']}


def family(kernel):
    role = kernel.get('role', '')
    if 'ffn_gate' in role and 'ffn_up' in role:
        return 'gate_up'
    for pattern, name in [('ffn_gate', 'gate'), ('ffn_up', 'up'), ('ffn_down', 'down'),
                          ('attn_q', 'Q'), ('attn_k', 'K'), ('attn_v', 'V'), ('attn_output', 'O')]:
        if pattern in role:
            return name
    if role == 'output.weight':
        return 'LM head'
    op = kernel.get('operation', {}).get('op', '')
    if op:
        return op
    return re.sub(r'blk\.\d+\.', '', role) or 'unattributed'


def resources(kernel):
    values = {}
    for key in ('vgpr_count', 'agpr_count', 'sgpr_count', 'lds_bytes', 'scratch_bytes'):
        if kernel.get(key) not in (None, ''):
            values[key] = finite(kernel[key])
    if 'vgpr_count' in values and 'agpr_count' in values:
        allocated = 8 * math.ceil((values['vgpr_count'] + values['agpr_count']) / 8)
        waves = min(8, 512 // allocated) if allocated else 8
        values.update(vector_registers_allocated=allocated, register_wave_limit_per_simd=waves)
        shape = str(kernel.get('workgroup', '')).split('x')
        if shape and all(s.isdigit() for s in shape):
            threads = math.prod(map(int, shape))
            if threads:
                values['register_workgroup_bound_per_cu'] = 4 * waves // math.ceil(threads / 64)
    return values


def values_for(kernel, kind):
    m = kernel['metrics']

    def get(name):
        if name in kernel.get('ambiguous_metrics', []) or m.get(name) is None:
            raise ValueError('Missing or ambiguous counter: ' + name)
        return finite(m[name])

    if kind == 'trace':
        return dict(duration_us=finite(kernel['duration_ns']) / 1000, **resources(kernel))
    if kind == 'memory':
        return {'duration_us': finite(kernel['duration_ns']) / 1000,
                'hbm_read_mb': get('FETCH_SIZE') * 1024 / 1e6,
                'hbm_write_mb': get('WRITE_SIZE') * 1024 / 1e6}
    if kind == 'attribution':
        total, fma, cvt = get('SQ_INSTS'), get('SQ_INSTS_VALU_FMA_F32'), get('SQ_INSTS_VALU_CVT')
        other = total - fma - cvt
        if other < 0:
            raise ValueError('Instruction categories exceed total issued instructions')
        return dict(instructions=total, fma_f32=fma, conversions=cvt, other_instructions=other)
    if kind == 'ipc':
        cycles = get('SQ_BUSY_CU_CYCLES')
        if cycles <= 0:
            raise ValueError('Zero active-cycle denominator')
        return {'ipc': get('SQ_INSTS') / cycles}
    if kind == 'activity':
        values = {'valu_busy_percent': get('VALU_BUSY_PERCENT'), 'mfma_busy_percent': get('MFMA_BUSY_PERCENT')}
        if any(v > 100.01 for v in values.values()):
            raise ValueError('Activity exceeds 100%; verify SDK counter definitions/device partition')
        return values
    if kind == 'occupancy':
        mean = get('MeanOccupancyPerActiveCU')
        if mean > 32.001:
            raise ValueError('Mean occupancy exceeds gfx942 32 waves/CU')
        return dict(mean_waves_per_active_cu=mean, occupancy_percent=100 * mean / 32)
    if kind == 'scratch':
        return dict(scratch_read_instructions=get('TA_BUFFER_READ_WAVEFRONTS_sum'),
                    scratch_write_instructions=get('TA_BUFFER_WRITE_WAVEFRONTS_sum'))
    if kind in ('mfma_i8', 'mfma_f16', 'scalar_fp32'):
        return coordinates(kernel, kind)
    raise ValueError('Unsupported comparison metric group: ' + kind)


def configuration_key(kernel):
    op = kernel['operation']
    return tuple(str(v) for v in (kernel['kernel'], kernel['grid'], kernel['workgroup'], op.get('type'),
                                 op.get('fusion', 'none'), op.get('matrix_count', '1')))


def gate_candidates(kernels, quant, phase, n):
    tensor = {'Q2_K': 'Q2_K', 'Q4_K_M': 'Q4_K'}[quant]
    result = []
    for k in kernels:
        op = k['operation']
        if (k['phase'] == phase and 'mul_mat' in k['kernel'] and family(k) in ('gate', 'up', 'gate_up') and
                (str(op.get('m')), str(op.get('n')), str(op.get('k'))) == ('28672', str(n), '8192') and
                all(t.upper() == tensor for t in op.get('type', '').split('+'))):
            result.append(k)
    return result


def select_samples(kernels, count, reference=None):
    groups = collections.defaultdict(list)
    for k in kernels:
        groups[configuration_key(k)].append(k)
    for key in sorted(groups):
        if reference is not None and key != reference:
            continue
        rows = sorted(groups[key], key=lambda k: (k['start_ns'], str(k['dispatch'])))
        if all(family(k) == 'gate_up' for k in rows):
            selected = rows[:count]
        else:
            # Odd sample counts retain one extra gate, matching the original
            # three-gate/two-up comparison when samples=5.
            gates = [k for k in rows if family(k) == 'gate'][:(count + 1) // 2]
            ups = [k for k in rows if family(k) == 'up'][:count // 2]
            if len(gates) != (count + 1) // 2 or len(ups) != count // 2:
                continue
            selected = gates + ups
        if len(selected) == count:
            return selected, key
    raise ValueError(f'No configuration has {count} matched gate/up samples (including required gate/up quotas)')


def analyze_captures(root, args):
    """Returns independently derived medians plus full dispatch provenance."""
    captures = collections.defaultdict(list)
    errors, selections, medians, operators, evidence = [], [], [], [], []
    directory = root / 'profiles' / args.profile_tag
    physical = None
    builds, protocols, fixtures = set(), set(), set()
    verified_cells = set()
    for path in sorted(directory.rglob('capture.json')):
        meta = read(path)
        cfg = meta['configuration']
        if (cfg['model_size'] != args.model_size or cfg['quant'] not in ('Q2_K', 'Q4_K_M') or
                cfg['prompt_tokens'] != args.bottleneck_context or cfg['concurrency'] != args.bottleneck_concurrency):
            continue
        phase, kind = meta.get('capture_phase'), meta['kind']
        if meta.get('backend') != 'rocm' or phase not in ('prefill', 'decode'):
            errors.append(f'{path}: expected a phase-gated AMD bottlenecks capture')
            continue
        if meta['status'] != 'captured':
            errors.append(f'{path}: {meta["status"]}')
            continue
        source = meta.get('source_cell')
        if source and not Path(source).is_file():
            pieces = Path(source).parts
            if 'cells' in pieces:
                source = str(root.joinpath(*pieces[pieces.index('cells'):]))
        if not source or not Path(source).is_file() or sha(source) != meta.get('source_cell_sha256'):
            errors.append(f'{path}: missing or changed source serving cell')
            continue
        if read(source)['configuration'] != cfg:
            errors.append(f'{path}: capture configuration differs from its serving fixture')
            continue
        if source not in verified_cells:
            try:
                validate_serving(Path(source))
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(f'{path}: source serving validation failed: {error}')
                continue
            verified_cells.add(source)
        if cfg['ubatch'] != args.ubatch or cfg['output_tokens'] != args.output_tokens or cfg['kv_type'] != 'f16':
            errors.append(f'{path}: incompatible batch/output/KV protocol')
            continue
        expected = cfg['devices']
        if physical is not None and physical != expected:
            raise ValueError('Publication cannot combine different HIP device selections')
        physical = expected
        builds.add(json.dumps(meta['build'], sort_keys=True))
        protocols.add(json.dumps(meta.get('protocol'), sort_keys=True))
        fixtures.add(meta.get('prompts_sha256'))
        if meta.get('captured_batches') != meta.get('capture_batches'):
            errors.append(f'{path}: batch gate did not complete')
            continue
        data_path = path.parent / 'kernels.json'
        if (not data_path.is_file() or sha(data_path) != meta.get('kernels_sha256') or not meta.get('raw_files') or
                any(not (path.parent / f).is_file() or sha(path.parent / f) != digest for f, digest in meta.get('raw_files', {}).items())):
            errors.append(f'{path}: raw or normalized counter evidence missing/changed')
            continue
        data = read(data_path)
        mapping = collections.defaultdict(set)
        for kernel in data['kernels']:
            if kernel.get('hip_device') is not None:
                mapping[kernel['agent']].add(str(kernel['hip_device']))
        if any(len(v) != 1 for v in mapping.values()):
            errors.append(f'{path}: profiler agent maps to more than one HIP device')
            continue
        evidence.append({'capture': str(path), 'sha256': sha(path), 'kernels_sha256': sha(data_path),
                         'source_cell_sha256': meta.get('source_cell_sha256'), 'coverage': data['coverage']})
        for k in data['kernels']:
            logical = k.get('hip_device')
            if logical is None or not str(logical).isdigit() or int(logical) >= len(expected):
                continue
            k = dict(k, source=str(path), physical_gpu=expected[int(logical)], quant=cfg['quant'])
            captures[(cfg['quant'], phase, int(logical), kind)].append(k)
    if len(builds) > 1 or len(protocols) > 1 or len(fixtures) > 1:
        raise ValueError('Publication captures use different diagnostic builds, protocols or prompt fixtures')
    physical = physical or args.devices
    for phase in ('prefill', 'decode'):
        scope = None
        for gpu in range(len(physical)):
            for quant in ('Q4_K_M', 'Q2_K'):
                n = args.ubatch if phase == 'prefill' else args.bottleneck_concurrency
                try:
                    reference_rows, reference = select_samples(gate_candidates(captures[(quant, phase, gpu, 'memory')], quant, phase, n), args.samples)
                    this_scope = reference[-2:]
                    if scope is not None and scope != this_scope:
                        raise ValueError('Q2/Q4 or GPUs have different fusion scopes; do not compare their kernel totals')
                    scope = this_scope
                except ValueError as error:
                    errors.append(f'{quant}/{phase}/GPU{physical[gpu]}: {error}')
                    continue
                for kind in ('trace', *BOTTLE_GROUPS):
                    try:
                        rows, _ = select_samples(gate_candidates(captures[(quant, phase, gpu, kind)], quant, phase, n), args.samples, reference)
                        if len({k['source'] for k in rows}) != 1:
                            raise ValueError('Duplicate captures for one format/phase/GPU/group')
                        values = [values_for(k, kind) for k in rows]
                        numeric = set.intersection(*(set(v) for v in values)) - {'definition'}
                        aggregated = {key: statistics.median(v[key] for v in values) for key in sorted(numeric)}
                        if kind in ('mfma_i8', 'mfma_f16', 'scalar_fp32'):
                            peak = {'mfma_i8': args.peak_int8_tops, 'mfma_f16': args.peak_mfma_tflops,
                                    'scalar_fp32': args.peak_fp32_tflops}[kind]
                            aggregated['work_percent_of_peak'] = 100 * aggregated['operations_per_s'] / (peak * 1e12)
                            aggregated['hbm_percent_of_peak'] = 100 * aggregated['dram_bytes_per_s'] / (args.bandwidth_tb_s * 1e12)
                        if kind == 'trace' and 'register_wave_limit_per_simd' not in aggregated:
                            raise ValueError('Missing VGPR/AGPR metadata; a newer rocprofv3 export is required')
                        if kind == 'attribution' and not math.isclose(aggregated['instructions'], sum(aggregated[k] for k in ('fma_f32', 'conversions', 'other_instructions')), rel_tol=1e-9):
                            raise ValueError('Median instruction categories do not reconcile; inspect individual launches')
                        medians.append(dict(quant=quant, phase=phase, gpu=physical[gpu], group=kind, samples=len(rows),
                                            m=28672, n=n, k=8192, fusion=reference[-2], matrix_count=reference[-1], **aggregated))
                        for row in rows:
                            selections.append(dict(quant=quant, phase=phase, gpu=physical[gpu], group=kind,
                                                   capture=row['source'], file=row['file'], process=row['process'], agent=row['agent'],
                                                   dispatch=row['dispatch'], kernel=row['kernel'], role=row['role'],
                                                   configuration=reference, **values_for(row, kind)))
                    except (ValueError, KeyError, TypeError) as error:
                        errors.append(f'{quant}/{phase}/GPU{physical[gpu]}/{kind}: {error}')
    # Operator coordinates keep separate geometries, fusion scopes, kernels and
    # launch configurations; no joining of counters from different passes.
    for (quant, phase, gpu, kind), rows in sorted(captures.items()):
        if kind != 'scalar_fp32':
            continue
        groups = collections.defaultdict(list)
        for row in rows:
            if row['phase'] != phase or family(row) in ('unattributed', 'unknown_fusion', 'gate', 'up', 'gate_up'):
                continue
            op = row['operation']
            key = (family(row), *configuration_key(row), op.get('m', ''), op.get('n', ''), op.get('k', ''))
            groups[key].append(row)
        for key, matching in groups.items():
            samples = sorted(matching, key=lambda r: r['start_ns'])
            try:
                values, used = [], []
                for k in samples:
                    # A cache-resident dispatch can have zero HBM traffic. Its
                    # counters remain valid, but no finite HBM roofline point.
                    if k['metrics'].get('FETCH_SIZE') == 0 and k['metrics'].get('WRITE_SIZE') == 0:
                        continue
                    values.append(coordinates(k, kind))
                    used.append(k)
                    if len(values) == args.samples:
                        break
                if not values or not any(v['operations'] > 0 for v in values):
                    continue
                operators.append(dict(quant=quant, phase=phase, gpu=physical[gpu], operator=key[0], configuration=key,
                                      samples=len(values), operations_per_byte=statistics.median(v['operations_per_byte'] for v in values),
                                      operations_per_s=statistics.median(v['operations_per_s'] for v in values),
                                      provenance=[{'capture': k['source'], 'agent': k['agent'], 'dispatch': k['dispatch']} for k in used]))
            except ValueError as error:
                errors.append(f'Operator {quant}/{phase}/GPU{physical[gpu]}/{key[0]}: {error}')
    coverage = []
    for (quant, phase, gpu, kind), rows in sorted(captures.items()):
        if kind != 'trace':
            continue
        def key(k):
            op = k['operation']
            return (family(k), *configuration_key(k), *(str(op.get(d, '')) for d in ('m', 'n', 'k', 'd0', 'd1', 'd2', 'd3')))
        reference = {key(k) for k in rows if k['phase'] == phase and family(k) not in ('unattributed', 'unknown_fusion')}
        counter = collections.defaultdict(list)
        for k in captures[(quant, phase, gpu, 'scalar_fp32')]:
            if k['phase'] == phase:
                counter[key(k)].append(k)
        for config in sorted(reference):
            valid, problems, zero_traffic = [], [], 0
            for k in counter[config]:
                try:
                    needed = ('FETCH_SIZE', 'WRITE_SIZE', 'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32', 'SQ_INSTS_VALU_FMA_F32')
                    if (all(k['metrics'].get(n) is not None and n not in k.get('ambiguous_metrics', []) for n in needed) and
                            k['duration_ns'] > 0 and k['metrics']['FETCH_SIZE'] == k['metrics']['WRITE_SIZE'] == 0):
                        zero_traffic += 1
                        continue
                    valid.append(coordinates(k, 'scalar_fp32'))
                except ValueError as error:
                    problems.append(str(error))
            # Zero traffic is a legitimate L2-resident dispatch but cannot have
            # an HBM roofline coordinate. Retain it as unavailable, never infinity.
            state = 'measured' if valid else 'zero_hbm_traffic' if zero_traffic else 'missing_or_undefined'
            if not valid and not zero_traffic:
                errors.append(f'{quant}/{phase}/GPU{physical[gpu]}/{config[0]}: missing or undefined FP32/HBM coordinates')
            coverage.append(dict(quant=quant, phase=phase, gpu=physical[gpu], configuration=config,
                                 status=state, valid_samples=len(valid), zero_traffic_samples=zero_traffic,
                                 errors=sorted(set(problems))))
    if not operators:
        errors.append('No other-operator scalar FP32 roofline coordinates')
    return {'medians': medians, 'selections': selections, 'operators': operators, 'errors': errors,
            'physical_gpus': physical, 'evidence': evidence, 'operator_coverage': coverage, 'complete': not errors}
