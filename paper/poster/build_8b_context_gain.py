#!/usr/bin/env python3
"""Plot measured 8B Q4_K_M throughput advantage over FP16 at fixed concurrency."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import build

OUT = build.HERE / 'figures' / 'llama-8b'
PROMPTS = (2048, 4096, 8192, 16384)


def main():
    build.theme()
    data = build.Data()
    series = []
    for concurrency in (8, 16, 32):
        q4, fp16 = [], []
        for prompt in PROMPTS:
            for quant, values in [('Q4_K_M', q4), ('FP16', fp16)]:
                row = data.get('8b', quant, concurrency, prompt)
                raw = json.loads((build.ROOT / row['raw_directory'] / 'r1/raw.json').read_text())['summary']
                assert raw['successful_requests'] == 2 * concurrency and raw['failed_requests'] == 0
                assert raw['max_client_inflight'] == concurrency and raw['cached_prompt_tokens'] == 0
                assert raw['mean_prompt_tokens'] == prompt and raw['mean_completion_tokens'] == 512
                assert math.isclose(row['output_tok_s'], raw['output_tokens_per_second'], rel_tol=1e-12)
                values.append(row['output_tok_s'])
        series.append({'concurrency': concurrency, 'context_tokens': list(PROMPTS),
                       'q4_throughput_tok_s': q4, 'fp16_throughput_tok_s': fp16,
                       'gain_pct': [100 * (q / f - 1) for q, f in zip(q4, fp16)]})
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=.12, right=.96, bottom=.21, top=.79)
    build.decorate(ax, 'Longer context reduces quantization gains',
                   'Llama 8B · Q4_K_M throughput advantage over FP16')
    for s, color, marker, style in zip(series, ['#7143a5', '#c66b12', build.Q4], ['o', '^', 's'], ['-', '--', ':']):
        values = s['gain_pct']
        ax.plot(range(4), values, marker=marker, linestyle=style, color=color,
                linewidth=2.4, markersize=7, label=f"Concurrency = {s['concurrency']}")
        for x, value in enumerate(values):
            ax.annotate(f'{value:+.1f}%', (x, value), xytext=(0, 9), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, color=color, fontweight='bold')
    ax.set_xticks(range(4), ['2K', '4K', '8K', '16K'])
    ax.set_xlim(-.15, 3.15)
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_formatter(build.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel('Context length per request', fontsize=14, color='#111111')
    ax.set_ylabel('Throughput advantage over FP16 (%)', fontsize=14, color='#111111')
    ax.tick_params(axis='both', labelcolor='#111111')
    ax.legend(loc='upper right', fontsize=10.5)
    fig.text(.12, .105, 'Gain = 100 × (Q4_K_M throughput / FP16 throughput − 1); matched context and concurrency.',
             fontsize=9, color=build.MUTED)
    fig.text(.12, .03, build.FOOTER, fontsize=9, color=build.MUTED, linespacing=1.5)
    build.save(fig, OUT / 'context-quantization-gain-8b')
    (OUT / 'context-quantization-gain-8b-data.json').write_text(json.dumps({
        'model': 'Llama 3.1 8B', 'formula': '100 * (Q4_K_M throughput / FP16 throughput - 1)',
        'series': series, 'selected_measured_rows': list(data.used.values()),
        'missing_values': [], 'validation': 'All 24 settings checked against raw summaries.'}, indent=2) + '\n')
    print('Wrote separate 8B context-gain PDF, PNG, SVG and source data.')


if __name__ == '__main__':
    main()
