"""Validate reuse when a context study adds formats and reduces run count.

The original measurement documents and producer fingerprints stay unchanged.
Only the repetition budget and additional model entries may differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def transition_errors(previous, current, previous_config, current_config):
    errors = []
    before, after = copy.deepcopy(previous), copy.deepcopy(current)
    old_runs, new_runs = before.pop("repetitions", None), after.pop("repetitions", None)
    if not isinstance(old_runs, int) or new_runs != 1 or old_runs < new_runs:
        errors.append("Only a reduction to one measured run is compatible")
    old_models = {entry["quant"]: entry for entry in before.pop("models", [])}
    new_models = {entry["quant"]: entry for entry in after.pop("models", [])}
    if not old_models or any(new_models.get(quant) != entry for quant, entry in old_models.items()):
        errors.append("Existing model identities changed or disappeared")
    for manifest, count in ((before, old_runs), (after, new_runs)):
        if manifest.get("configuration", {}).pop("REPETITIONS", None) != str(count):
            errors.append("Configuration and repetition budget disagree")
    old_code, new_code = before.pop("code_sha256", {}), after.pop("code_sha256", {})
    if set(old_code) != set(new_code):
        errors.append("Producer dependency paths changed")
    for path in old_code:
        if old_code[path] == new_code.get(path):
            continue
        if not path.endswith("/config/context-study.env"):
            errors.append("Measurement source changed: " + path)
            continue
        for contents, expected in ((previous_config, old_code[path]), (current_config, new_code.get(path))):
            if not isinstance(contents, str) or hashlib.sha256(contents.encode()).hexdigest() != expected:
                errors.append("Configuration source proof does not match its hash")
        normalize = lambda text: re.sub(r'^REPETITIONS=.*$', 'REPETITIONS="1"', text or "", flags=re.M)
        if normalize(previous_config) != normalize(current_config):
            errors.append("Configuration changed beyond repetition count")
    if before != after:
        errors.append("Workload, hardware, backend or runtime controls changed")
    return errors


def producer_manifest(current, history, producer_fingerprint):
    if producer_fingerprint == fingerprint(current):
        return current
    if history.get("current_fingerprint") != fingerprint(current) or history.get("selected_repetition") != 1:
        return None
    producer = history.get("producers", {}).get(producer_fingerprint, {})
    previous = producer.get("manifest", {})
    if fingerprint(previous) != producer_fingerprint:
        return None
    if transition_errors(previous, current, producer.get("configuration_source"), history.get("configuration_source")):
        return None
    return previous
