#!/usr/bin/env python3
"""Llama 70B on two RTX PRO 6000 GPUs: 4 formats × C8/16/32 × 2K–32K.

All inputs use C warmup and 2C measured requests, 512 outputs and FP16 KV.
Use build, prepare, plan, run, report in that order; selectors apply to plan/run.
"""
import subprocess
import sys

from run_a100_serving import main

if __name__ == "__main__":
    try:
        main(default_hardware="rtxpro6000")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
