#!/usr/bin/env python3
"""Separate MI300X entry point: setup, prepare, plan, serve, profile, compute, report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from mi300x.common import FORMATS, ROOT, STATE, lock


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['setup', 'prepare', 'plan', 'serve', 'doctor', 'profile', 'bottlenecks', 'compute', 'report', 'paper'])
    p.add_argument('--rocm', type=Path, default=Path('/opt/rocm'))
    p.add_argument('--devices', nargs='+', default=['0'], help='Physical HIP indices; e.g. --devices 0 1')
    p.add_argument('--split', nargs='+', type=float, help='Layer split weights; default equal across selected GPUs')
    p.add_argument('--model-size', choices=['1b', '8b', '70b'], default='70b')
    p.add_argument('--formats', nargs='+', choices=FORMATS, default=list(FORMATS))
    p.add_argument('--manifest', type=Path)
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models/mi300x')
    p.add_argument('--contexts', nargs='+', type=int, default=[2048, 4096, 8192, 16384, 65536])
    p.add_argument('--concurrency', nargs='+', type=int, default=[8, 16, 32])
    p.add_argument('--output-tokens', type=int, default=512)
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--ubatch', type=int, default=512)
    p.add_argument('--headroom-gib', type=float, default=2)
    p.add_argument('--output', type=Path, default=ROOT / 'results/mi300x/70b-context')
    p.add_argument('--port', type=int, help='Defaults: inference 18080, profiler 18081')
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--startup-timeout', type=float, default=1800)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--diagnostic', action='store_true', help='setup only: build separate annotated HIP server and gate')
    p.add_argument('--resume', action='store_true', help='Keep only completed cells with identical configuration/build/GPU identity')
    p.add_argument('--cells', type=Path, nargs='+', help='profile: select saved AMD cell.json files/directories explicitly')
    p.add_argument('--profile-tag', default='run1', help='New tag required to repeat captures without overwriting')
    from mi300x.profiling import GROUPS
    p.add_argument('--groups', nargs='*', choices=GROUPS, default=list(GROUPS), help='Empty list gives trace-only profiling')
    p.add_argument('--kernel-regex', default='.*', help='rocprofv3 kernel include regex')
    p.add_argument('--launch-skip', type=int, default=0)
    p.add_argument('--launch-count', type=int, default=8, help='Counter dispatch cap per matching kernel name; trace is uncapped')
    p.add_argument('--capture-phase', choices=['prefill', 'decode'], help='profile: gate pure batches of this phase')
    p.add_argument('--capture-batches', type=int, default=1, help='Number of pure batches per bottlenecks capture')
    p.add_argument('--counter-uncapped', action='store_true', help='profile: collect all dispatches in the gated region')
    p.add_argument('--kernel-regex-compute', default='mul_mat', help='rocprof-compute substring filter, not a regex')
    p.add_argument('--compute-timeout', type=float, default=86400)
    p.add_argument('--no-plots', action='store_true', help='report: export CSV/JSON using only Python standard library')
    p.add_argument('--peak-fp32-tflops', type=float, default=163.4)
    p.add_argument('--peak-mfma-tflops', type=float, default=1307.4)
    p.add_argument('--peak-int8-tops', type=float, default=2614.9)
    p.add_argument('--bandwidth-tb-s', type=float, default=5.3)
    p.add_argument('--samples', type=int, default=5, help='paper: matched launches per GPU/phase/configuration')
    p.add_argument('--allow-incomplete', action='store_true', help='paper: export an explicitly incomplete snapshot without failing')
    p.add_argument('--bottleneck-context', type=int, default=2048, help='paper: context of Q2/Q4 detailed comparison')
    p.add_argument('--bottleneck-concurrency', type=int, default=8, help='paper: concurrency of Q2/Q4 detailed comparison')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.action == 'paper' and args.model_size != '70b':
        p.error('The publication gate/up geometry targets 70B; use report for other model sizes')
    if not all(d.isdigit() for d in args.devices) or len(set(args.devices)) != len(args.devices):
        p.error('--devices must be distinct nonnegative HIP indices')
    args.split = args.split or [1.] * len(args.devices)
    if len(args.split) != len(args.devices) or any(not (0 < x < float('inf')) for x in args.split):
        p.error('--split must contain one finite positive weight per GPU')
    for name in ('contexts', 'concurrency'):
        values = getattr(args, name)
        if any(x <= 0 for x in values) or len(values) != len(set(values)):
            p.error(f'--{name} values must be distinct and positive')
    for name in ('output_tokens', 'repetitions', 'batch', 'ubatch', 'timeout', 'startup_timeout', 'jobs',
                 'headroom_gib', 'launch_count', 'compute_timeout', 'peak_fp32_tflops', 'peak_mfma_tflops', 'peak_int8_tops',
                 'bandwidth_tb_s', 'capture_batches', 'samples', 'bottleneck_context', 'bottleneck_concurrency'):
        if not (0 < getattr(args, name) < float('inf')):
            p.error(f'--{name.replace("_", "-")} must be finite and positive')
    if args.launch_skip < 0 or args.ubatch > args.batch:
        p.error('Require launch-skip >= 0 and ubatch <= batch')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.profile_tag):
        p.error('--profile-tag must use letters, numbers, underscores or hyphens')
    if len(set(args.formats)) != len(args.formats) or len(set(args.groups)) != len(args.groups):
        p.error('Duplicate format or counter group')
    args.port = args.port or (18081 if args.action in ('profile', 'bottlenecks') else 18080)
    if not 1024 <= args.port <= 65535:
        p.error('--port must be between 1024 and 65535')
    args.output, args.model_dir, args.rocm = args.output.resolve(), args.model_dir.resolve(), args.rocm.resolve()
    try:
        if args.action == 'plan':
            from mi300x.serving import grid
            print(json.dumps(grid(args), indent=2))
            return 0
        with lock():
            if args.action == 'setup':
                from mi300x.setup import build
                build(args, args.diagnostic)
            elif args.action == 'prepare':
                from mi300x.setup import prepare
                prepare(args)
            elif args.action == 'serve':
                from mi300x.serving import serve
                failures = serve(args)
                if failures:
                    print(f'{len(failures)} serving cells failed; per-cell errors were saved.', file=sys.stderr)
                    return 2
            elif args.action in ('doctor', 'profile', 'bottlenecks', 'compute'):
                from mi300x import profiling
                result = getattr(profiling, args.action)(args)
                if args.action == 'doctor':
                    print(json.dumps(result, indent=2))
                if args.action in ('profile', 'bottlenecks') and result:
                    print(f'{len(result)} captures failed or have unsupported counter groups; inspect capture.json.', file=sys.stderr)
                    return 2
            elif args.action == 'report':
                from mi300x.report import report
                report(args)
            elif args.action == 'paper':
                from mi300x.publication import build
                complete = build(args)
                if not complete and not args.allow_incomplete:
                    print(f'AMD publication coverage is incomplete; inspect publication/{args.profile_tag}/validation.json.', file=sys.stderr)
                    return 2
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(f'MI300X: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('MI300X run interrupted; owned child processes were stopped.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
