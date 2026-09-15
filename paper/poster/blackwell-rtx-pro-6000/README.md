# RTX PRO 6000 Blackwell runtime-cost poster

This is an **incomplete snapshot** built from 36 validated measurements, 4 capacity exclusions, and 20 unresolved cells in the requested 60-cell grid.

The unresolved cells are left blank and are not interpolated or plotted as zero. The committed inputs contain validated report CSVs and audits but not the raw cell trees, so cross-server prompt hashes and the precise GPU edition could not be independently rechecked here.

## Sources

- `results/cuda-context-study-rtxpro6000-iq1-q8-r1/serving-report/`
- `results/cuda-context-study-rtxpro6000-q2-q4-r1/serving-report/`

## Outputs

- `runtime-cost-poster.pdf`
- `runtime-cost-poster.png`
- `runtime-cost-poster.svg`
- `figures/` with each panel in PDF, PNG, and SVG
- `coverage.csv`, `data.json`, and `evidence.json`

## Rebuild

```bash
python3 paper/poster/build.py --serving-reports \
  results/cuda-context-study-rtxpro6000-iq1-q8-r1 \
  results/cuda-context-study-rtxpro6000-q2-q4-r1 \
  --allow-missing --output paper/poster/blackwell-rtx-pro-6000
```
