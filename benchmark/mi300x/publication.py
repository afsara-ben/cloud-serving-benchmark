"""AMD-only equivalents of the serving poster and runtime bottlenecks report.

Reads only the marked MI300X study. Does not import the CUDA paper builders.
"""
from __future__ import annotations

import collections
import itertools
import math
import statistics
import textwrap

from .analysis import analyze_captures, delta, validate_serving
from .common import COMMIT, output_root, read, write
from .metrics import definitions
from .report import PEAK_SOURCE, table

COLORS = {'IQ1_M': '#8b5fbf', 'Q2_K': '#d47916', 'Q4_K_M': '#247a49', 'Q8_0': '#3479aa'}
EXCLUDED = {'capacity_estimated', 'observed_oom', 'observed_headroom_limit', 'native_context_unsupported'}


def percent(value):
    return 'N/A' if value is None else f'{value:+.1f}%'


def context_label(value):
    return f'{value // 1024}K' if value % 1024 == 0 else str(value)


def serving_data(root, args):
    rows, errors = {}, []
    fixtures, hardware = {}, None
    evidence = []
    for path in sorted((root / 'cells').rglob('cell.json')):
        cell = read(path)
        cfg = cell['configuration']
        if (cfg['model_size'] != args.model_size or cfg['quant'] not in args.formats or
                cfg['prompt_tokens'] not in args.contexts or cfg['concurrency'] not in args.concurrency):
            continue
        if cell.get('backend') != 'rocm':
            raise ValueError('CUDA data found in an AMD study')
        key = (cfg['quant'], cfg['concurrency'], cfg['prompt_tokens'])
        point = rows.setdefault(key, {'runs': [], 'statuses': [], 'sources': []})
        point['statuses'].append(cell['status'])
        point['sources'].append(str(path))
        if cell['status'] != 'completed':
            continue
        try:
            run = validate_serving(path)
            if cfg['output_tokens'] != args.output_tokens or cfg['kv_type'] != 'f16':
                raise ValueError('Output length / KV precision differs from requested publication protocol')
            if hardware is not None and run['gpu_identity'] != hardware:
                raise ValueError('Different GPU identities in one poster')
            hardware = run['gpu_identity']
            fixture_key = (cfg['concurrency'], cfg['prompt_tokens'], cell['repetition'])
            if fixture_key in fixtures and fixtures[fixture_key] != run['prompt_hashes']:
                raise ValueError('Prompts differ across quantization formats')
            fixtures[fixture_key] = run['prompt_hashes']
            point['runs'].append(run)
            evidence.append(run)
        except (ValueError, KeyError, OSError, TypeError) as error:
            point['statuses'][-1] = 'invalid'
            errors.append(f'{path}: {error}')
    coverage = []
    for quant, c, p in itertools.product(args.formats, args.concurrency, args.contexts):
        point = rows.get((quant, c, p), {'runs': [], 'statuses': [], 'sources': []})
        runs = point['runs']
        statuses = point['statuses']
        if runs and len(runs) == args.repetitions and len(statuses) == len(runs):
            status = 'completed'
        elif statuses and all(s in EXCLUDED for s in statuses) and len(statuses) == args.repetitions:
            status = 'excluded'
        else:
            status = 'incomplete'
            errors.append(f'{quant}/P{p}/C{c}: expected {args.repetitions} validated runs or documented capacity exclusions')
        record = {'quant': quant, 'concurrency': c, 'prompt_tokens': p, 'status': status,
                  'repetitions': len(runs), 'recorded_statuses': statuses, 'sources': point['sources']}
        if runs:
            for metric in ('ttft_p95_ms', 'tpot_p95_ms', 'output_tokens_per_second'):
                record[metric] = statistics.median(r[metric] for r in runs)
        coverage.append(record)
    return {'coverage': coverage, 'errors': errors, 'evidence': evidence,
            'gpu_identity': hardware, 'complete': not errors}


def poster_figures(directory, data, args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    rows = {(r['quant'], r['concurrency'], r['prompt_tokens']): r for r in data['coverage']}
    prompts = sorted(args.contexts)
    x = np.arange(len(prompts))

    def value(q, c, p, metric='output_tokens_per_second'):
        # Missing values are NaN, so lines break rather than bridge excluded cells.
        return rows.get((q, c, p), {}).get(metric, float('nan'))

    def f1(ax):
        quants = [q for q in ('Q4_K_M', 'Q2_K', 'IQ1_M') if q in args.formats]
        width = .8 / max(1, len(quants))
        for i, q in enumerate(quants):
            ys = [value(q, 8, p, 'ttft_p95_ms') / 1000 for p in prompts]
            bars = ax.bar(x + (i - (len(quants) - 1) / 2) * width, ys, width,
                          color=COLORS[q], label=q)
            for p, b, y in zip(prompts, bars, ys):
                if not math.isfinite(y):
                    continue
                label = f'{y:.1f}'
                reference = value('Q4_K_M', 8, p, 'ttft_p95_ms') / 1000
                if q != 'Q4_K_M' and math.isfinite(reference):
                    label += '\n' + percent(delta(reference, y))
                ax.annotate(label, (b.get_x() + b.get_width() / 2, y), xytext=(0, 3),
                            textcoords='offset points', ha='center', fontsize=7)
        ax.set(title='F1  Quantization and first-token latency', ylabel='p95 TTFT (s)',
               xticks=x, xticklabels=[context_label(p) for p in prompts], xlabel='Input tokens · C8')
        ax.margins(y=.32); ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.02))

    def f2(ax):
        styles = [('o', '-'), ('^', '--'), ('s', ':'), ('D', '-.'), ('x', '-')]
        for q in args.formats:
            for i, p in enumerate(prompts):
                ys = [value(q, c, p) for c in sorted(args.concurrency)]
                if any(math.isfinite(y) for y in ys):
                    marker, style = styles[i % len(styles)]
                    ax.plot(sorted(args.concurrency), ys, marker=marker, linestyle=style, color=COLORS[q],
                            label=f'{q} / {context_label(p)}', markersize=4)
        ax.set(title='F2  Concurrency and throughput', xlabel='Concurrent clients', ylabel='Output tokens/s',
               xticks=sorted(args.concurrency), ylim=(0, None))
        if ax.lines:
            handles = [Line2D([], [], color=COLORS[q], label=q) for q in args.formats]
            handles += [Line2D([], [], color='#555555', marker=styles[i % len(styles)][0],
                               linestyle=styles[i % len(styles)][1], label=context_label(p)) for i, p in enumerate(prompts)]
            ax.legend(handles=handles, fontsize=7, ncol=3, loc='upper left')

    def f3(ax):
        quants = [q for q in ('IQ1_M', 'Q2_K', 'Q4_K_M') if q in args.formats]
        width = .8 / max(1, 2 * len(quants))
        for i, q in enumerate(quants):
            for j, c in enumerate((8, 16)):
                offset = (2 * i + j - (2 * len(quants) - 1) / 2) * width
                ys = [value(q, c, p) for p in prompts]
                bars = ax.bar(x + offset, ys, width, color=COLORS[q], hatch='///' if c == 8 else None,
                              edgecolor='white', linewidth=.5, label=f'{q} C{c}')
                for b, y in zip(bars, ys):
                    if math.isfinite(y):
                        ax.annotate(f'{y:.1f}', (b.get_x() + width / 2, y), xytext=(0, 2),
                                    textcoords='offset points', ha='center', fontsize=6, rotation=90)
            for index, p in enumerate(prompts):
                a, b = value(q, 8, p), value(q, 16, p)
                if math.isfinite(a) and math.isfinite(b):
                    ax.annotate(percent(delta(a, b)), (index + (2 * i + .5 - (2 * len(quants) - 1) / 2) * width, max(a, b)),
                                xytext=(0, 26 + 10 * (i % 2)), textcoords='offset points', ha='center', fontsize=6)
        ax.set(title='F3  Throughput at 8 and 16 clients', ylabel='Output tokens/s',
               xticks=x, xticklabels=[context_label(p) for p in prompts], xlabel='Input tokens')
        ax.margins(y=.5); ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.02))

    def f4(ax):
        for i, p in enumerate(prompts):
            a, b = value('Q4_K_M', 8, p), value('IQ1_M', 8, p)
            for q, y in [('Q4_K_M', a), ('IQ1_M', b)]:
                if math.isfinite(y):
                    ax.scatter(y, i, color=COLORS[q], s=40, label=q if i == 0 else None, zorder=3)
            if math.isfinite(a) and math.isfinite(b):
                ax.plot([a, b], [i, i], color='#81909a', zorder=2)
                ax.annotate(percent(delta(a, b)), (max(a, b), i), xytext=(7, 0),
                            textcoords='offset points', va='center', fontsize=9)
        ax.set(title='F4  IQ1_M relative to Q4_K_M', xlabel='Output tokens/s · C8',
               yticks=x, yticklabels=[context_label(p) for p in prompts], ylabel='Input tokens')
        ax.invert_yaxis(); ax.margins(x=.4, y=.2); ax.legend(fontsize=8)

    state = 'validated' if data['complete'] else 'INCOMPLETE SNAPSHOT'
    gpu_count = len(data['gpu_identity'] or args.devices)
    footer = (f'{gpu_count} × MI300X · {args.output_tokens} outputs/request · FP16 KV · uncached inputs · '
              f'{args.repetitions} run(s)/setting\nMissing/capacity-excluded values have no point; see coverage.csv. '
              'Ratios compare measurements, not statistical significance.')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for draw, ax in zip((f1, f2, f3, f4), axes.flat):
        draw(ax); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle(f'MI300X runtime cost · {args.model_size} · {state}', fontsize=17)
    fig.text(.5, .016, footer, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=3, w_pad=2)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(directory / f'runtime-cost-poster.{suffix}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    f1(axes[0])
    for q in args.formats:
        for i, c in enumerate(sorted(args.concurrency)):
            ys = [value(q, c, p) for p in prompts]
            if any(math.isfinite(y) for y in ys):
                axes[1].plot(prompts, ys, marker=['o', '^', 's', 'D'][i % 4], linestyle=['-', '--', ':', '-.'][i % 4],
                             color=COLORS[q], label=f'{q} C{c}')
    axes[1].set(title='Throughput across contexts', xscale='log', xlabel='Input tokens', ylabel='Output tokens/s',
                xticks=prompts, xticklabels=[context_label(p) for p in prompts], ylim=(0, None))
    if axes[1].lines:
        axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f'MI300X runtime cost · {args.model_size} · {state}')
    fig.text(.5, .012, footer, ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(directory / 'runtime-cost-merged.pdf'); plt.close(fig)


def comparisons(data, args):
    result = []
    lookup = {(r['quant'], r['phase'], r['gpu'], r['group']): r for r in data['medians']}
    for phase, gpu, group in sorted({(r['phase'], r['gpu'], r['group']) for r in data['medians']}):
        a, b = (lookup.get((q, phase, gpu, group), {}) for q in ('Q4_K_M', 'Q2_K'))
        for metric in sorted(set(a) | set(b)):
            if metric in ('quant', 'phase', 'gpu', 'group', 'samples', 'm', 'n', 'k', 'fusion', 'matrix_count'):
                continue
            reference, value = a.get(metric), b.get(metric)
            result.append(dict(phase=phase, gpu=gpu, group=group, metric=metric, Q4_K_M=reference, Q2_K=value,
                               change_percent=delta(reference, value),
                               samples_q4=a.get('samples', 0), samples_q2=b.get('samples', 0)))
    shares = []
    for phase, gpu in itertools.product(('prefill', 'decode'), data['physical_gpus']):
        a, b = (lookup.get((q, phase, gpu, 'attribution'), {}) for q in ('Q4_K_M', 'Q2_K'))
        if a and b:
            total = b['instructions'] - a['instructions']
            extra = b['fma_f32'] + b['conversions'] - a['fma_f32'] - a['conversions']
            shares.append(dict(phase=phase, gpu=gpu, extra_instructions=total, extra_fma_and_conversion=extra,
                               share_percent=100 * extra / total if total > 0 else None))
    return result, shares


def bottleneck_figures(directory, data, args, comparison, shares):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    lookup = {(r['phase'], r['gpu'], r['group'], r['metric']): r for r in comparison}
    gpu = data['physical_gpus'][0]
    state = 'validated' if data['complete'] else 'INCOMPLETE: see validation.json'
    note = (f'{args.model_size} · MI300X GPU {gpu} · P={args.bottleneck_context}, C={args.bottleneck_concurrency} · '
            f'{args.samples} launches/configuration · {state}')

    def save(pdf, fig, stem):
        fig.text(.5, .015, note, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94), h_pad=2.5)
        pdf.savefig(fig)
        fig.savefig(directory / 'figures' / (stem + '.png'), dpi=160)
        fig.savefig(directory / 'figures' / (stem + '.pdf'))
        plt.close(fig)

    def bar(ax, phase, group, metric, title, divisor=1, unit=''):
        row = lookup.get((phase, gpu, group, metric), {})
        for x, q in enumerate(('Q4_K_M', 'Q2_K')):
            y = row.get(q)
            if y is None:
                ax.text(x, .05, 'N/A', transform=ax.get_xaxis_transform(), ha='center')
            else:
                y /= divisor
                ax.bar(x, y, color=COLORS[q], width=.55)
                label = f'{y:,.3g}'
                if q == 'Q2_K':
                    label += '\n' + percent(row.get('change_percent'))
                ax.annotate(label, (x, y), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set(title=title, ylabel=unit, xticks=[0, 1], xticklabels=['Q4', 'Q2'], xlim=(-.7, 1.7))
        ax.margins(y=.35); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)

    def roof_axis(ax, kind, phase, peak):
        xs = np.logspace(-3, 5, 400)
        ax.loglog(xs, np.minimum(xs * args.bandwidth_tb_s, peak), '--', color='gray', label='Per-GPU rated ceiling')
        found = False
        for q in ('Q4_K_M', 'Q2_K'):
            for r in data['medians']:
                if (r['quant'], r['phase'], r['gpu'], r['group']) != (q, phase, gpu, kind):
                    continue
                if r['operations_per_s'] > 0 and r['operations_per_byte'] > 0:
                    ax.scatter(r['operations_per_byte'], r['operations_per_s'] / 1e12,
                               color=COLORS[q], s=70, marker='o' if q == 'Q4_K_M' else 'D', label=q, zorder=3)
                    found = True
        if not found:
            ax.text(.5, .4, 'No validated nonzero work in this arithmetic domain', transform=ax.transAxes, ha='center', fontsize=9)
        ax.set(xlabel='Issued operations / HBM byte', ylabel='Issued TOP/s', title=f'{phase.title()} · {kind}')
        ax.grid(alpha=.2); ax.legend(fontsize=8)

    with PdfPages(directory / 'runtime-bottlenecks-report.pdf') as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        fig.suptitle('Prefill gate/up: instruction work, pipeline activity and memory', fontsize=16)
        specs = [('ipc', 'ipc', 'IPC (AMD convention)', 1, 'Issued instructions / active CU cycles'),
                 ('activity', 'valu_busy_percent', 'VALU busy', 1, '%'),
                 ('activity', 'mfma_busy_percent', 'MFMA busy', 1, '%'),
                 ('attribution', 'instructions', 'Total issued instructions', 1e6, 'Million wave instructions'),
                 ('memory', 'hbm_read_mb', 'HBM reads', 1, 'MB (decimal)'),
                 ('memory', 'duration_us', 'Kernel duration', 1000, 'ms')]
        for ax, spec in zip(axes.flat, specs):
            bar(ax, 'prefill', *spec)
        save(pdf, fig, 'prefill-actual-values')
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        fig.suptitle('Decode gate/up: register demand, residency and work', fontsize=16)
        specs = [('trace', 'vector_registers_allocated', 'Allocated VGPR + AGPR', 1, 'Registers/lane'),
                 ('trace', 'register_wave_limit_per_simd', 'Vector-register-only residency bound', 1, 'Wave64 / SIMD'),
                 ('occupancy', 'occupancy_percent', 'Achieved occupancy (active CUs)', 1, '% of 32 waves/CU'),
                 ('attribution', 'instructions', 'Total issued instructions', 1e6, 'Million wave instructions'),
                 ('memory', 'hbm_read_mb', 'HBM reads', 1, 'MB (decimal)'),
                 ('memory', 'duration_us', 'Kernel duration', 1, 'µs')]
        for ax, spec in zip(axes.flat, specs):
            bar(ax, 'decode', *spec)
        save(pdf, fig, 'decode-register-pressure')
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        fig.suptitle('Issued F32 FMA and type-conversion instructions', fontsize=16)
        selected_shares = {s['phase']: s['share_percent'] for s in shares if s['gpu'] == gpu}
        fig.text(.5, .945, 'FMA + conversion share of positive net extra instructions: ' +
                 ' · '.join(f'{p}: {percent(selected_shares.get(p))}' for p in ('prefill', 'decode')),
                 ha='center', fontsize=9)
        for i, phase in enumerate(('prefill', 'decode')):
            for ax, metric, title in zip(axes[i], ('fma_f32', 'conversions', 'other_instructions'),
                                        ('F32 FMA/MAD', 'All VALU conversions', 'Other issued instructions')):
                bar(ax, phase, 'attribution', metric, f'{phase.title()} · {title}', 1e6, 'Million wave instructions')
        save(pdf, fig, 'instruction-overhead-actual-values')
        fig = plt.figure(figsize=(11.7, 8.3))
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.4])
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        fig.suptitle('Matrix arithmetic rooflines: INT8 and F16 stay separate', fontsize=15)
        roof_axis(axes[0], 'mfma_i8', 'prefill', args.peak_int8_tops)
        roof_axis(axes[1], 'mfma_f16', 'prefill', args.peak_mfma_tflops)
        table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off')
        cells = []
        for metric, label, divisor in [('operations_per_byte', 'INT8 operations / HBM byte', 1),
                                       ('operations_per_s', 'INT8 issued TOP/s', 1e12),
                                       ('work_percent_of_peak', 'INT8 work / rated peak (%)', 1),
                                       ('dram_bytes_per_s', 'HBM read + write GB/s', 1e9),
                                       ('hbm_percent_of_peak', 'HBM traffic / rated peak (%)', 1)]:
            row = lookup.get(('prefill', gpu, 'mfma_i8', metric), {})
            cells.append([label, *['N/A' if row.get(q) is None else f'{row[q]/divisor:,.3f}' for q in ('Q4_K_M', 'Q2_K')],
                          percent(row.get('change_percent'))])
        t = table_ax.table(cellText=cells, colLabels=['Prefill INT8 path', 'Q4', 'Q2', 'Q2 change'],
                           cellLoc='center', loc='center', colWidths=[.46, .18, .18, .18])
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
        save(pdf, fig, 'prefill-matrix-roofline')
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 9))
        fig.suptitle('Other operators: FP32 issued-wave proxy (assumes 64 active lanes)', fontsize=14)
        ops = sorted({p['operator'] for p in data['operators']})
        palette = plt.get_cmap('tab20')
        for ax, phase in zip(axes, ('prefill', 'decode')):
            xs = np.logspace(-3, 5, 300)
            ax.loglog(xs, np.minimum(xs * args.bandwidth_tb_s, args.peak_fp32_tflops), '--', color='gray')
            for i, op in enumerate(ops):
                for q in ('Q4_K_M', 'Q2_K'):
                    points = [p for p in data['operators'] if p['phase'] == phase and p['operator'] == op and p['quant'] == q]
                    if points:
                        ax.scatter([p['operations_per_byte'] for p in points], [p['operations_per_s'] / 1e12 for p in points],
                                   edgecolors=[palette(i % 20)], facecolors=[palette(i % 20)] if q == 'Q4_K_M' else 'none',
                                   marker='o', s=40, label=f'{op} / {q}')
            ax.set(title=f'{phase.title()} · selected GPUs shown individually; Q4 filled, Q2 hollow',
                   xlabel='Proxy FP32 operations / HBM byte', ylabel='Proxy TFLOP/s')
            ax.grid(alpha=.2)
            if ax.collections:
                ax.legend(fontsize=5, ncol=4, loc='upper left')
        save(pdf, fig, 'operator-roofline-combined')
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        fig.suptitle('Decode scratch activity and register allocation', fontsize=16)
        specs = [('scratch', 'scratch_read_instructions', 'Spill/stack read instructions', 1, 'Issued wave instructions'),
                 ('scratch', 'scratch_write_instructions', 'Spill/stack write instructions', 1, 'Issued wave instructions'),
                 ('trace', 'scratch_bytes', 'Scratch allocation', 1, 'Bytes/work-item'),
                 ('trace', 'register_workgroup_bound_per_cu', 'Vector-register-only workgroup bound', 1, 'Workgroups / CU (upper bound)')]
        for ax, spec in zip(axes.flat, specs):
            bar(ax, 'decode', *spec)
        save(pdf, fig, 'scratch-and-residency')
        fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis('off')
        fig.suptitle('Selected GPUs: matched gate/up comparisons', fontsize=16)
        lines = [['Phase', 'GPU', 'Q4 time µs', 'Q2 time µs', 'Time change', 'Instructions', 'HBM reads']]
        for phase, g in itertools.product(('prefill', 'decode'), data['physical_gpus']):
            row = lookup.get((phase, g, 'memory', 'duration_us'), {})
            fmt = lambda v: 'N/A' if v is None else f'{v:,.3f}'
            lines.append([phase, g, fmt(row.get('Q4_K_M')), fmt(row.get('Q2_K')), percent(row.get('change_percent')),
                          percent(lookup.get((phase, g, 'attribution', 'instructions'), {}).get('change_percent')),
                          percent(lookup.get((phase, g, 'memory', 'hbm_read_mb'), {}).get('change_percent'))])
        t = ax.table(cellText=lines[1:], colLabels=lines[0], loc='center', cellLoc='center')
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.3)
        save(pdf, fig, 'both-gpus')
        fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis('off')
        fig.suptitle('Measurement definitions, attribution and coverage', fontsize=16)
        paragraphs = [
            'Each hardware pass captures complete pure prefill or full-concurrency decode batches. Gate/up comparisons select the same M/N/K, tensor recipe and fusion scope, with one kernel/grid/workgroup configuration per format. The default five samples mean three gate plus two up launches, or five fused gate/up launches when both formats use that fusion.',
            'Instruction counters count issued wave instructions. The conversion category covers all VALU type conversions, not just integer-to-float. Shares below describe additional instructions, not their share of execution time. The AMD compiler and kernels can change the direction of a quantization comparison.',
            'The pinned GGUF format definitions use 16 weights per scale/offset group for Q2_K and 32 for Q4_K: respectively 16 and 8 groups per block of 256 weights. The report validates the selected tensor types before comparing them. This layout fact does not establish an AMD runtime slowdown.',
            'IPC follows ROCm Compute Profiler gfx942 definitions. Occupancy uses 32 wave64 slots/CU and may be inaccurate for short kernels. Register bounds use the combined VGPR/AGPR allocation and omit other residency limits. Scratch instruction counts include stack accesses and cannot isolate register spills.',
            'Roofline operands come from the same dispatch and pass. INT8 and F16 MFMA work assumes full EXEC. FP32 uses 64*(ADD+MUL+2*FMA), an issued-wave proxy. Rated ceilings are per GPU; no cross-GPU timing or work is summed. Profiler timing is excluded from serving figures.',
            'Single-capture launch medians do not estimate run-to-run uncertainty or prove a causal bottleneck. All values and sample identities are in the adjacent CSV/JSON files. Missing measurements remain unavailable.',
        ]
        for s in shares:
            paragraphs.append(f'{s["phase"].title()}, GPU {s["gpu"]}: F32 FMA plus conversion share of the positive net instruction increase = {percent(s["share_percent"])}.')
        if data['errors']:
            paragraphs.append(f'INCOMPLETE: {len(data["errors"])} validation issues. See validation.json for exact missing cases and reasons.')
        for x, section in [(.02, paragraphs[:3]), (.53, paragraphs[3:])]:
            ax.text(x, .98, '\n\n'.join(textwrap.fill(p, 62) for p in section), va='top', fontsize=9.5, linespacing=1.3)
        save(pdf, fig, 'method-and-coverage')


def build(args):
    root = output_root(args.output)
    directory = root / 'publication' / args.profile_tag
    (directory / 'figures').mkdir(parents=True, exist_ok=True)
    poster = serving_data(root, args)
    kernels = analyze_captures(root, args)
    comparison, shares = comparisons(kernels, args)
    complete = poster['complete'] and kernels['complete']
    # Save evidence and validation before rendering, even for incomplete runs.
    write(directory / 'poster-data.json', poster)
    write(directory / 'report-data.json', kernels)
    write(directory / 'validation.json', {'complete': complete, 'poster_complete': poster['complete'],
                                          'bottlenecks_complete': kernels['complete'],
                                          'poster_errors': poster['errors'], 'bottleneck_errors': kernels['errors']})
    write(directory / 'definitions.json', {**definitions(), 'peak_source': PEAK_SOURCE,
          'quantization_layout': {'Q2_K': {'weights_per_group': 16, 'groups_per_block': 16},
                                  'Q4_K': {'weights_per_group': 32, 'groups_per_block': 8},
                                  'source': f'https://github.com/ggml-org/llama.cpp/blob/{COMMIT}/ggml/src/ggml-common.h'},
          'rated_peaks_per_gpu': {'int8_tops': args.peak_int8_tops, 'f16_tflops': args.peak_mfma_tflops,
                                 'fp32_tflops': args.peak_fp32_tflops, 'hbm_tb_per_s': args.bandwidth_tb_s}})
    for name, rows in [('coverage', poster['coverage']), ('comparison-data', comparison),
                       ('instruction-attribution-data', shares), ('selected-kernels', kernels['selections']),
                       ('operator-roofline-data', kernels['operators']), ('kernel-medians', kernels['medians'])]:
        table(directory / (name + '.csv'), rows)
    table(directory / 'operator-coverage.csv', kernels['operator_coverage'])
    write(directory / 'evidence.json', {'serving': poster['evidence'], 'profiling': kernels['evidence']})
    text = ('# MI300X runtime measurements\n\n' + ('Validated coverage.' if complete else '**Incomplete measurement coverage.**') +
            '\n\nThe serving poster uses validated unprofiled requests. The bottleneck report uses pure-phase HIP captures. '
            'Counter domains, normalization and sources are in definitions.json; validation.json lists missing evidence.\n\n'
            'The AMD instruction attribution uses F32 FMA and all VALU conversions. It does not reproduce NVIDIA SASS opcode counts. '
            'No direction of slowdown, bandwidth limitation or instruction overhead is assumed before measurement.\n')
    (directory / 'README.md').write_text(text)
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                             'axes.spines.top': False, 'axes.spines.right': False})
        poster_figures(directory, poster, args)
        bottleneck_figures(directory, kernels, args, comparison, shares)
    print(f'MI300X publication: {directory} ({"complete" if complete else "incomplete"})', flush=True)
    return complete
