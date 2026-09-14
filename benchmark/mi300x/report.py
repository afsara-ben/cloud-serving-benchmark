from __future__ import annotations

import bisect
import collections
import csv
import math
from pathlib import Path
import statistics

from .common import output_root, read, write
from .profiling import csv_rows, number
from .metrics import canonical

PEAK_SOURCE = 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf'
COUNTER_SOURCE = 'https://rocm.docs.amd.com/en/docs-6.3.1/conceptual/gpu-arch/mi300-mi200-performance-counters.html'


def labels(text):
    return dict(piece.split('=', 1) for piece in text.split(':')[1:] if '=' in piece)


class Ranges:
    """Interval lookup with prefix maxima; accommodates nested CPU launch ranges."""
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: int(r['Start_Timestamp']))
        self.starts = [int(r['Start_Timestamp']) for r in self.rows]
        self.ends = []
        maximum = 0
        for row in self.rows:
            maximum = max(maximum, int(row['End_Timestamp']))
            self.ends.append(maximum)

    def at(self, timestamp):
        i = bisect.bisect_right(self.starts, timestamp) - 1
        result = []
        while i >= 0 and self.ends[i] >= timestamp:
            row = self.rows[i]
            if int(row['End_Timestamp']) >= timestamp:
                result.append(row)
            i -= 1
        return sorted(result, key=lambda r: int(r['End_Timestamp']) - int(r['Start_Timestamp']))


def normalize_capture(directory):
    """Use correlation IDs, never GPU timestamps inside asynchronous CPU ranges.

    A call processes ONE capture/pass. No cross-pass counter joins are allowed.
    Ambiguous counter dimensions remain unavailable, with the raw rows retained.
    """
    root = directory / 'rocprof'
    default_pid = (directory / 'gate.ready').read_text().strip() if (directory / 'gate.ready').exists() else 'target'

    def pid(row):
        return row.get('Process_Id') or default_pid

    markers = collections.defaultdict(list)
    apis = collections.defaultdict(list)
    for path in root.rglob('*marker_api_trace.csv'):
        for r in csv_rows(path):
            label = r.get('Message') or r.get('Function') or r.get('Name', '')
            if label.startswith(('csb_op:', 'csb_batch:')):
                markers[(pid(r), r.get('Thread_Id', ''))].append(dict(r, label=label))
    ranges = {key: Ranges(rows) for key, rows in markers.items()}
    for path in root.rglob('*hip_api_trace.csv'):
        for r in csv_rows(path):
            if r.get('Correlation_Id'):
                apis[(pid(r), r['Correlation_Id'])].append(r)
    counter_files = sorted(root.rglob('*counter_collection.csv'))
    trace_metadata = collections.defaultdict(list)
    for path in root.rglob('*kernel_trace.csv'):
        for r in csv_rows(path):
            if r.get('Kernel_Name'):
                trace_metadata[(pid(r), r['Agent_Id'], r['Dispatch_Id'])].append(r)
    files = counter_files or sorted(root.rglob('*kernel_trace.csv'))
    kernels = {}
    for path in files:
        for r in csv_rows(path):
            if not r.get('Kernel_Name'):
                continue
            # Each rocprof invocation has one counter group. The file is part of
            # the identity so repeated dispatch IDs in another pass cannot join.
            key = (str(path.relative_to(root)), pid(r), r['Agent_Id'], r['Dispatch_Id'])
            start, end = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            if end <= start:
                raise ValueError(f'Invalid dispatch timestamps: {key}')
            if key not in kernels:
                traces = trace_metadata.get((pid(r), r['Agent_Id'], r['Dispatch_Id']), [])
                # CSV counter exports can flatten XYZ and omit some launch
                # resource fields. Enrich only through the same dispatch in
                # this invocation's trace, never through another replay pass.
                trace = traces[0] if len(traces) == 1 and traces[0]['Kernel_Name'] == r['Kernel_Name'] else {}
                host = apis.get((pid(r), r.get('Correlation_Id', '')), [])
                host = [h for h in host if 'launch' in (h.get('Function') or h.get('Name', '')).lower()]
                scopes = []
                if len(host) == 1:
                    h = host[0]
                    index = ranges.get((pid(h), h.get('Thread_Id', '')))
                    scopes = index.at(int(h['Start_Timestamp'])) if index else []
                op = next((labels(s['label']) for s in scopes if s['label'].startswith('csb_op:')), {})
                batch = next((labels(s['label']) for s in scopes if s['label'].startswith('csb_batch:')), {})

                def shape(name):
                    source = trace if trace else r
                    return source.get(name) or 'x'.join(source.get(name + '_' + axis, '?') for axis in 'XYZ')

                def resource(name):
                    return r.get(name) or trace.get(name)

                kernels[key] = {'file': key[0], 'process': key[1], 'agent': key[2], 'dispatch': key[3],
                                'kernel': r['Kernel_Name'], 'duration_ns': end - start,
                                'start_ns': start, 'end_ns': end, 'role': op.get('role', 'unattributed'),
                                'phase': batch.get('phase', 'unknown'), 'operation': op, 'batch': batch,
                                'grid': shape('Grid_Size'), 'workgroup': shape('Workgroup_Size'),
                                'vgpr_count': resource('VGPR_Count'), 'sgpr_count': resource('SGPR_Count'),
                                'agpr_count': resource('Accum_VGPR_Count'), 'hip_device': op.get('hip_device'),
                                'lds_bytes': resource('LDS_Block_Size'), 'scratch_bytes': resource('Scratch_Size'),
                                'metrics': {}, 'ambiguous_metrics': []}
            kernel = kernels[key]
            name = canonical(r.get('Counter_Name', ''))
            if name:
                if name in kernel['metrics']:
                    kernel['ambiguous_metrics'].append(name)
                    kernel['metrics'][name] = None
                else:
                    try:
                        kernel['metrics'][name] = number(r.get('Counter_Value'))
                    except (TypeError, ValueError):
                        kernel['metrics'][name] = None
    rows = list(kernels.values())
    agents = [dict(r, file=str(path.relative_to(root))) for path in root.rglob('*agent_info.csv') for r in csv_rows(path)]
    return {'backend': 'rocm', 'kernels': rows, 'agent_inventory': agents,
            'agent_identity': 'Raw ROCprofiler Agent_Id; not assumed equal to HIP logical device index.',
            'coverage': {'dispatches': len(rows), 'with_operator_and_phase': sum(
                k['role'] not in ('unattributed', 'unknown_fusion') and k['phase'] != 'unknown' for k in rows),
                'ambiguous_counter_dispatches': sum(bool(k['ambiguous_metrics']) for k in rows)}}


def coordinates(kernel, kind):
    m = kernel['metrics']

    def value(name):
        if name in kernel['ambiguous_metrics'] or m.get(name) is None:
            raise ValueError(f'Missing/ambiguous {name}')
        return number(m[name])

    # FETCH_SIZE and WRITE_SIZE are derived KiB counters in the SDK catalogue.
    traffic = 1024 * (value('FETCH_SIZE') + value('WRITE_SIZE'))
    duration = kernel['duration_ns'] * 1e-9
    if traffic <= 0 or duration <= 0:
        raise ValueError('Nonpositive duration or memory traffic')
    if kind == 'scalar_fp32':
        work = 64 * (value('SQ_INSTS_VALU_ADD_F32') + value('SQ_INSTS_VALU_MUL_F32') +
                     2 * value('SQ_INSTS_VALU_FMA_F32'))
        definition = 'Wave64-scaled FP32 ADD/MUL/FMA proxy; does not count active EXEC lanes. Excludes MFMA and transcendental instructions.'
    elif kind in ('mfma_f16', 'mfma_i8'):
        precision = 'F16' if kind == 'mfma_f16' else 'I8'
        work = 512 * value('SQ_INSTS_VALU_MFMA_MOPS_' + precision)
        definition = f'Issued {precision} MFMA operations (512 operations per counter unit, assumes full EXEC); excludes other arithmetic domains.'
    else:
        raise ValueError('Unsupported roofline arithmetic domain')
    return {'operations': work, 'dram_bytes': traffic, 'duration_s': duration,
            'operations_per_byte': work / traffic, 'operations_per_s': work / duration,
            'dram_bytes_per_s': traffic / duration,
            'definition': definition}


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        fields = list(dict.fromkeys(key for r in rows for key in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    root = output_root(args.output)
    report_dir = root / 'reports'
    report_dir.mkdir(exist_ok=True)
    serving = []
    captures, kernels, points = [], [], []
    for path in sorted((root / 'cells').rglob('cell.json')):
        cell = read(path)
        if cell.get('backend') != 'rocm':
            raise ValueError(f'Non-AMD cell found in AMD output: {path}')
        cfg = cell['configuration']
        serving.append({'model_size': cfg['model_size'], 'quant': cfg['quant'], 'prompt_tokens': cfg['prompt_tokens'],
                        'concurrency': cfg['concurrency'], 'repetition': cell['repetition'],
                        'gpu_count': len(cfg['devices']), 'status': cell['status'], 'cell': str(path),
                        **(cell.get('summary', {}) if cell['status'] == 'completed' else {})})
    for path in sorted((root / 'profiles').rglob('capture.json')):
        capture = read(path)
        if capture.get('backend') != 'rocm':
            raise ValueError(f'Non-AMD capture found: {path}')
        cfg = capture['configuration']
        identity = {'model_size': cfg['model_size'], 'quant': cfg['quant'], 'prompt_tokens': cfg['prompt_tokens'],
                    'concurrency': cfg['concurrency'], 'capture': str(path.parent), 'kind': capture['kind']}
        status = dict(identity, status=capture['status'], error=capture.get('error', ''))
        if capture['status'] == 'captured':
            data = read(path.parent / 'kernels.json')
            status.update(data['coverage'])
            for k in data['kernels']:
                row = dict(identity, agent=k['agent'], phase=k['phase'], role=k['role'], kernel=k['kernel'],
                           dispatch=k['dispatch'], duration_ns=k['duration_ns'], grid=k['grid'], workgroup=k['workgroup'],
                           vgpr_count=k['vgpr_count'], agpr_count=k.get('agpr_count'), hip_device=k.get('hip_device'),
                           sgpr_count=k['sgpr_count'], lds_bytes=k['lds_bytes'],
                           scratch_bytes=k['scratch_bytes'], tensor_type=k['operation'].get('type'),
                           m=k['operation'].get('m'), n=k['operation'].get('n'), k=k['operation'].get('k'),
                           fusion=k['operation'].get('fusion'), batch_width=k['batch'].get('width'),
                           prefill_tokens=k['batch'].get('prefill_tokens'), decode_tokens=k['batch'].get('decode_tokens'))
                row.update(k['metrics'])
                kernels.append(row)
                if capture['kind'] in ('scalar_fp32', 'mfma_f16', 'mfma_i8'):
                    try:
                        values = coordinates(k, capture['kind'])
                    except ValueError:
                        continue
                    points.append(dict(row, **values))
        captures.append(status)
    table(report_dir / 'serving.csv', serving)
    table(report_dir / 'capture-coverage.csv', captures)
    table(report_dir / 'kernel-values.csv', kernels)
    table(report_dir / 'roofline-values.csv', points)
    write(report_dir / 'definitions.json', {
        'backend': 'rocm', 'peaks_per_gpu': {'fp32_vector_tflops': args.peak_fp32_tflops,
            'mfma_f16_tflops': args.peak_mfma_tflops, 'mfma_i8_tops': args.peak_int8_tops, 'hbm_tb_per_s': args.bandwidth_tb_s},
        'peak_source': PEAK_SOURCE, 'counter_source': COUNTER_SOURCE,
        'peaks': 'Theoretical default ceilings; override with measured ceilings if available.',
        'counter_join': 'Operands from the same capture, counter file, process, agent and dispatch only.',
        'scalar_fp32': '64*(ADD_F32+MUL_F32+2*FMA_F32): issued-wave proxy, not predicated-on thread FLOPs.',
        'mfma_f16': '512*SQ_INSTS_VALU_MFMA_MOPS_F16; separate domain from scalar/vector FP32.',
        'mfma_i8': '512*SQ_INSTS_VALU_MFMA_MOPS_I8; separate integer arithmetic domain.',
        'serving': 'Median of per-run p95 latencies / per-run throughput; uncached exact inputs and fixed output.',
        'coverage': 'Counter launch cap applies per matching kernel name; it does not guarantee coverage of every operator or phase.',
        'timing': 'Kernel times may overlap across queues/GPUs; sums are not wall time. Profile timings never enter serving figures.'})
    if not args.no_plots:
        plots(report_dir, serving, points, args)
    print(f'MI300X reports: {report_dir}', flush=True)


def plots(directory, serving, points, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    colors = {'IQ1_M': '#8c4ca3', 'Q2_K': '#d47916', 'Q4_K_M': '#247a49', 'Q8_0': '#3479aa'}
    good = [r for r in serving if r['status'] == 'completed']
    # One PDF page per model size and request concurrency, two latency panels.
    if good:
        with PdfPages(directory / 'runtime-cost-merged.pdf') as pdf, PdfPages(directory / 'throughput-context.pdf') as tp:
            for size, c in sorted({(r['model_size'], r['concurrency']) for r in good}):
                selected = [r for r in good if r['model_size'] == size and r['concurrency'] == c]
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
                throughput_fig, ax_tp = plt.subplots(figsize=(6, 4), constrained_layout=True)
                for q in colors:
                    rows = [r for r in selected if r['quant'] == q]
                    xs = sorted({r['prompt_tokens'] for r in rows})
                    for ax, metric, label in [(axes[0], 'ttft_p95_ms', 'p95 TTFT (ms)'),
                                               (axes[1], 'tpot_p95_ms', 'p95 TPOT (ms)'),
                                               (ax_tp, 'output_tokens_per_second', 'Output tokens/s')]:
                        ys = [statistics.median(r[metric] for r in rows if r['prompt_tokens'] == x) for x in xs]
                        if xs:
                            ax.plot(xs, ys, 'o-', color=colors[q], label=q)
                        ax.set(xscale='log', xlabel='Input tokens', ylabel=label)
                        ax.grid(alpha=.2)
                axes[0].legend(fontsize=8)
                ax_tp.legend(fontsize=8)
                fig.suptitle(f'MI300X · {size} · {c} concurrent requests · unprofiled')
                ax_tp.set_title(f'MI300X · {size} · {c} concurrent requests')
                pdf.savefig(fig); tp.savefig(throughput_fig)
                plt.close(fig); plt.close(throughput_fig)
    if points:
        with PdfPages(directory / 'roofline-explained.pdf') as pdf:
            combinations = sorted({(p['model_size'], p['prompt_tokens'], p['concurrency'], p['phase']) for p in points})
            for size, prompt, c, phase in combinations:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
                for ax, kind, peak, title in [(axes[0], 'scalar_fp32', args.peak_fp32_tflops, 'FP32 issued-wave proxy'),
                                               (axes[1], 'mfma_f16', args.peak_mfma_tflops, 'F16 MFMA issued work'),
                                               (axes[2], 'mfma_i8', args.peak_int8_tops, 'INT8 MFMA issued work')]:
                    x = np.logspace(-3, 5, 300)
                    ax.loglog(x, np.minimum(peak * 1e12, x * args.bandwidth_tb_s * 1e12) / 1e12,
                              '--', color='gray', label='Per-GPU theoretical ceiling')
                    for q in colors:
                        rows = [p for p in points if (p['model_size'], p['prompt_tokens'], p['concurrency'], p['phase'], p['kind'], p['quant']) ==
                                (size, prompt, c, phase, kind, q) and p['operations'] > 0]
                        if rows:
                            ax.scatter([r['operations_per_byte'] for r in rows], [r['operations_per_s'] / 1e12 for r in rows],
                                       label=q, color=colors[q], s=12, alpha=.55)
                    ax.set(xlabel='Counted operations / HBM byte', ylabel='Counted TOP/s', title=title)
                    ax.grid(alpha=.2); ax.legend(fontsize=7)
                fig.suptitle(f'MI300X per-dispatch samples · {size} · P={prompt}, C={c} · {phase}\n'
                             'FP32 proxy assumes 64 active lanes; operands stay within one counter pass', fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)
