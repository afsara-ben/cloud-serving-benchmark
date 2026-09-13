# Short 70B paper

Read [llama70b.pdf](llama70b.pdf) or edit [llama70b.md](llama70b.md). The three-page paper follows the reference study's separation between **Runtime Cost** and **Runtime Bottlenecks**, using the completed 512-output 70B measurements. It contains two figures and two tables.

Build from the repository root:

```sh
python3 paper/build.py
```

Dependencies: `matplotlib`, `markdown`, and `reportlab`. This workspace keeps additional document dependencies in `.run/paper-deps`, which the builder discovers automatically; use `/home/hys4qm/anaconda3/bin/python3` here. On another machine, install dependencies in a virtual environment.

The Markdown is the text source for the PDF. The builder reads existing result CSVs to generate both plots, saves vector PDF and PNG versions under `figures/`, and records input paths/hashes in [evidence.json](evidence.json). Tables and narrative are authored in Markdown and must be reviewed if the underlying study changes. No inference or profiling is launched.

The paper does not claim completion of the broader study, whole-model acceleration from the isolated dispatch control, or results for pending 70B formats. Full measurements and raw-counter links remain in the [focused report](../results/cuda-context-study/diagnostics/q2-q4-priority/README.md).

The [poster figures](poster/README.md) present four Runtime Cost findings using paired scatter groups, concurrency lines, gain bars, and dumbbells. They include a combined layout, individual PNG/PDF/SVG exports, and a separate, explicitly illustrative 64K extrapolation.

An [alternate two-line PDF](poster/runtime-cost-two-line.pdf) combines the same four findings into FP16-normalized throughput and concurrency-gain plots, showing all five formats on 8B with a separately labeled 70B latency callout.
