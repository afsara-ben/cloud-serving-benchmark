#!/usr/bin/env python3
"""Check the 70B posters against their retained request records."""
import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POSTER = Path(__file__).resolve().parent


def key(row):
    return row["format"], row["clients"], row["input_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true", help="Validate an explicitly incomplete data snapshot.")
    args = parser.parse_args()
    merged = json.loads((POSTER / "merged-data.json").read_text())
    coverage = merged["coverage"]
    expected = {(q, c, p) for q in ("IQ1_M", "Q2_K", "Q4_K_M", "Q8_0")
                for c in (8, 16, 32) for p in (2048, 4096, 8192, 16384)}
    assert len(coverage) == 48 and {key(r) for r in coverage} == expected
    assert all(r["model"] == "70b" for r in coverage)
    counts = collections.Counter(r["status"] for r in coverage)
    allowed = {"complete", "capacity_estimated", "observed_oom", "observed_headroom_limit"}
    if args.allow_missing:
        allowed.add("missing")
    assert set(counts) <= allowed
    raw_by_key, signatures = {}, {}
    groups = collections.defaultdict(list)
    requests = 0
    for row in coverage:
        if row["status"] != "complete":
            continue
        q, c, p = key(row)
        raw = json.loads((ROOT / row["evidence"] / "r1/raw.json").read_text())
        raw_by_key[key(row)] = raw
        records = raw["requests"]
        assert len(records) == 2 * c
        assert raw["summary"]["max_client_inflight"] == c
        assert raw["summary"]["failed_requests"] == 0
        assert raw["summary"]["cached_prompt_tokens"] == 0
        assert all(r["ok"] and r["completion_tokens"] == 512 and r["prompt_tokens"] == p for r in records)
        signature = sorted((r["request_index"], r["input_sha256"]) for r in records)
        if (c, p) in signatures:
            assert signature == signatures[c, p], ("Input mismatch across formats", c, p, q)
        signatures[c, p] = signature
        groups[c, p].append(q)
        requests += len(records)

    assert {key(r) for r in merged["selected_measured_rows"]} == set(raw_by_key)
    for filename in ("merged-data.json", "data.json"):
        data = json.loads((POSTER / filename).read_text())
        assert data["coverage"] == coverage
        assert not data["main_poster_contains_synthetic_values"]
        for row in data["selected_measured_rows"]:
            assert row["model"] == "70b"
            summary = raw_by_key[key(row)]["summary"]
            for plotted, source, divisor in (("output_tok_s", "output_tokens_per_second", 1),
                                             ("ttft_p95_s", "ttft_p95_ms", 1000),
                                             ("tpot_p95_ms", "tpot_p95_ms", 1)):
                assert math.isclose(row[plotted], summary[source] / divisor, rel_tol=1e-12), (filename, key(row), plotted)

    report = {
        "status": "passed", "requested_settings": len(coverage), "statuses": dict(counts),
        "missing_measurements_allowed": args.allow_missing,
        "validated_measured_requests": requests, "output_tokens_per_request": 512,
        "matched_input_groups": [{"clients": c, "input_tokens": p, "formats": qs}
                                 for (c, p), qs in groups.items()],
        "checks": ["Exact prompt/output token counts", "All retained requests successful",
                   "Requested concurrency reached", "No cached prompt tokens",
                   "Identical request-index/input-hash pairs across quantizations at each setting",
                   "Both posters' throughput, TTFT and TPOT values equal their raw measurements",
                   "All 48 requested settings documented; only 70B data and no synthetic values"],
    }
    (POSTER / "measurement-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
