#!/usr/bin/env python3
"""Create separate measured Llama 8B F1/F3 plots without modifying 70B exports."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import build

OUT = build.HERE / 'figures' / 'llama-8b'
PROMPTS = (2048, 4096, 8192, 16384)
COLORS = {'IQ1_M': '#7143a5', 'Q2_K': build.Q2, 'Q4_K_M': build.Q4}


def main():
    build.theme()
    data = build.Data()
    rows = {}
    for quant in COLORS:
        for concurrency in (8, 16, 32):
            for prompt in PROMPTS:
                row = data.get('8b', quant, concurrency, prompt)
                raw_path = build.ROOT / row['raw_directory'] / 'r1' / 'raw.json'
                raw = json.loads(raw_path.read_text())
                summary = raw['summary']
                assert summary['successful_requests'] == 2 * concurrency
                assert summary['failed_requests'] == 0
                assert summary['max_client_inflight'] == concurrency
                assert summary['cached_prompt_tokens'] == 0
                assert summary['mean_prompt_tokens'] == prompt
                assert summary['mean_completion_tokens'] == 512
                assert math.isclose(row['output_tok_s'], summary['output_tokens_per_second'], rel_tol=1e-12)
                assert math.isclose(row['ttft_p95_s'], summary['ttft_p95_ms'] / 1000, rel_tol=1e-12)
                row = dict(row, raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
                rows[quant, concurrency, prompt] = row

    def canvas(title, subtitle, ylabel):
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.subplots_adjust(left=.12, right=.98, bottom=.19, top=.82)
        build.decorate(ax, title, subtitle)
        ax.set_xticks(range(4), ['2K', '4K', '8K', '16K'])
        ax.set_xlabel('Context length per request', fontsize=14, color='#111111')
        ax.set_ylabel(ylabel, fontsize=14, color='#111111')
        ax.tick_params(axis='both', labelcolor='#111111')
        fig.text(.12, .025, build.FOOTER, fontsize=8.5, color=build.MUTED, linespacing=1.5)
        return fig, ax

    f1_changes, f3_changes = [], []
    fig, ax = canvas('F1  Lower bits ≠ lower latency',
                     '8B · concurrency = 8 · percentages relative to Q4_K_M', 'TTFT p95 (s)')
    maximum = max(rows[q, 8, p]['ttft_p95_s'] for q in COLORS for p in PROMPTS)
    for quant, offset in zip(('IQ1_M', 'Q2_K', 'Q4_K_M'), (-.25, 0, .25)):
        values = [rows[quant, 8, p]['ttft_p95_s'] for p in PROMPTS]
        positions = [i + offset for i in range(4)]
        ax.bar(positions, values, width=.22, color=COLORS[quant], label=quant, zorder=3)
        for x, value in zip(positions, values):
            ax.text(x, value + maximum * .015, f'{value:.1f}', ha='center', va='bottom',
                    fontsize=9, color=COLORS[quant], fontweight='bold')
    for i, p in enumerate(PROMPTS):
        top = max(rows[q, 8, p]['ttft_p95_s'] for q in COLORS)
        for q, label, gap in [('Q2_K', 'Q2', .11), ('IQ1_M', 'IQ1', .19)]:
            change = 100 * (rows[q, 8, p]['ttft_p95_s'] / rows['Q4_K_M', 8, p]['ttft_p95_s'] - 1)
            ax.text(i, top + maximum * gap, f'{label} {change:+.1f}%', ha='center', va='bottom',
                    fontsize=9, color=COLORS[q], fontweight='bold')
            f1_changes.append({'format': q, 'context_tokens': p, 'ttft_change_pct': change})
    ax.set_ylim(0, maximum * 1.43)
    ax.legend(loc='upper left', ncol=3, fontsize=9.5)
    build.save(fig, OUT / 'f1-precision-latency-8b')

    fig, axes = plt.subplots(1, 3, figsize=(19, 8.5), sharey=False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.72, wspace=.14)
    fig.text(.065, .94, 'F3  Throughput versus concurrency', fontsize=23, fontweight='bold')
    fig.text(.065, .885, 'Llama 8B · point labels: tok/s · percentages: change relative to concurrency = 8',
             fontsize=12, color=build.MUTED)
    styles = [(2048, 'o', '-'), (4096, '^', '--'), (8192, 's', ':'), (16384, 'D', '-.')]
    fig.legend(handles=[build.Line2D([], [], color=build.INK, marker=m, linestyle=l,
                                    label=f'{p // 1024}K context') for p, m, l in styles],
               loc='upper left', bbox_to_anchor=(.065, .855), ncol=4, fontsize=11)
    maximum = max(r['output_tok_s'] for r in rows.values())
    f3_series = []
    for ax, (q, color) in zip(axes, COLORS.items()):
        ax.set_title(q, fontsize=16, color=color, fontweight='bold', pad=14)
        ax.grid(axis='y', color='#e5e9ee', linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xticks([8, 16, 32])
        ax.set_xlim(5, 35)
        ax.set_ylim(0, max(r['output_tok_s'] for r in rows.values() if r['format'] == q) * 1.18)
        ax.set_xlabel('Concurrency', fontsize=14, color='#111111')
        ax.tick_params(axis='both', labelcolor='#111111')
        for p, marker, line in styles:
            values = [rows[q, c, p]['output_tok_s'] for c in (8, 16, 32)]
            ax.plot([8, 16, 32], values, color=color, marker=marker, linestyle=line,
                    linewidth=2, markersize=6, markerfacecolor='white' if p == 16384 else color)
            f3_series.append({'format': q, 'context_tokens': p, 'concurrency': [8, 16, 32],
                              'throughput_tok_s': values})
            for concurrency, value in zip((8, 16, 32), values):
                label = f'{value:.1f}'
                if concurrency != 8:
                    change = 100 * (value / values[0] - 1)
                    label += f'\n({change:+.1f}%)'
                    f3_changes.append({'format': q, 'context_tokens': p, 'concurrency': concurrency,
                                       'reference_concurrency': 8, 'throughput_change_pct': change})
                ax.annotate(label, (concurrency, value), xytext=(0, 8), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    axes[0].set_ylabel('Aggregate output throughput (tok/s)', fontsize=14, color='#111111')
    fig.text(.065, .055, build.FOOTER, fontsize=10, color=build.MUTED, linespacing=1.5)
    build.save(fig, OUT / 'f3-concurrency-gain-8b')
    (OUT / 'data.json').write_text(json.dumps({'model': 'Llama 3.1 8B', 'missing_values': [],
        'selected_measured_rows': list(rows.values()), 'f1_changes': f1_changes, 'f3_changes': f3_changes,
        'f3_series': f3_series, 'f3_encoding': {'x': 'concurrency', 'panels': 'quantization', 'line_and_marker': 'context length'},
        'f1_formula': '100 * (TTFT_format / TTFT_Q4_K_M - 1), concurrency = 8',
        'f3_formula': '100 * (throughput_C / throughput_C8 - 1), C = 16 or 32',
        'validation': 'All 36 settings checked against raw summaries; 1344 successful requests.'}, indent=2) + '\n')
    print(f'Wrote separate 8B F1/F3 PDF, PNG, SVG and verified provenance to {OUT}')


if __name__ == '__main__':
    main()
