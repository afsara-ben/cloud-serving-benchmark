#!/usr/bin/env python3
"""Validate one RTX PRO 6000 shard and write small, Git-friendly result JSONs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import rtxpro6000_publication as publication


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")


def portable(value, study):
    """Replace checkout-specific absolute study paths throughout exported data."""
    if isinstance(value, dict):
        return {portable(key, study): portable(item, study) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, study) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(study):
                return str(Path("study") / path.relative_to(study))
        except (OSError, ValueError):
            pass
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study, captures = args.study_root.resolve(), args.captures.resolve()
    output = (args.output or ROOT / "results/rtxpro6000-exports" / study.name).resolve()
    output.mkdir(parents=True, exist_ok=True)

    serving = publication.serving_data(study)
    profile = publication.bottleneck_data(study, captures)
    if set(profile["formats"]) != set(serving["audit"]["scope"]["model_formats"]["70b"]):
        raise ValueError("Profile and serving shard formats differ")

    files = []
    for quant in profile["formats"]:
        serving_path = output / f"{quant}-serving-r1.json"
        full_path = output / f"{quant}-profile-full-sections-p{profile['prompt_tokens']}.json"
        scalar_path = output / f"{quant}-profile-scalar-roofline-p{profile['prompt_tokens']}.json"
        write(serving_path, portable({"schema_version": 1, "hardware": "rtxpro6000", "format": quant,
            "repetitions": 1, "rows": [row for row in serving["rows"] if row["format"] == quant],
            "coverage": [row for row in serving["coverage"] if row["format"] == quant],
            "devices": serving["devices"], "source_hashes": serving["sources"]}, study))
        write(full_path, portable({"schema_version": 1, "hardware": "rtxpro6000", "format": quant,
            "prompt_tokens": profile["prompt_tokens"], "analysis_scope": profile["analysis_scope"],
            "layer_counts": profile["layer_counts"], "cases": [row for row in profile["cases"] if row["format"] == quant],
            "metrics": [row for row in profile["metric_rows"] if row["format"] == quant],
            "instructions": [row for row in profile["instruction_rows"] if row["format"] == quant],
            "tensor_roofline": [row for row in profile["tensor_samples"] if row["format"] == quant],
            "layer_extrapolation": [row for row in profile["extrapolation_rows"] if row["format"] == quant]}, study))
        write(scalar_path, portable({"schema_version": 1, "hardware": "rtxpro6000", "format": quant,
            "prompt_tokens": profile["prompt_tokens"], "analysis_scope": profile["analysis_scope"],
            "roofline_plan": profile["operator_plan"], "audit": profile["operator_audit"],
            "points": [row for row in profile["operator_points"] if row["format"] == quant]}, study))
        files.extend((serving_path, full_path, scalar_path))
    capture_name = str(captures.relative_to(study)) if captures.is_relative_to(study) else str(captures)
    index = {"schema_version": 1, "study": study.name, "capture_directory": capture_name,
             "formats": profile["formats"], "repetitions": 1, "files": []}
    for path in files:
        index["files"].append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size})
    report = study / "paper" / f"runtime-bottlenecks-c8-p{profile['prompt_tokens']}" / "runtime-bottlenecks-report.pdf"
    if report.is_file():
        report_copy = output / f"{profile['formats'][0]}-{profile['formats'][1]}-runtime-bottlenecks-p{profile['prompt_tokens']}.pdf"
        shutil.copy2(report, report_copy)
        index["files"].append({"path": report_copy.name, "sha256": hashlib.sha256(report_copy.read_bytes()).hexdigest(),
                               "bytes": report_copy.stat().st_size})
    write(output / "index.json", index)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
