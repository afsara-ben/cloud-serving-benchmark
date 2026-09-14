"""CUDA study hardware identities and per-GPU rated scalar rooflines.

RTX specifications were checked against the linked NVIDIA product pages on
2026-09-14. These are rated ceilings, not measured sustained performance.
"""
from __future__ import annotations

import re

HARDWARE = ("a100", "rtxpro6000")


def matches(name, hardware):
    if hardware == "a100":
        return "A100" in name
    if hardware == "rtxpro6000":
        return bool(re.search(r"RTX PRO 6000.*Blackwell", name, re.I))
    raise ValueError(f"Unknown hardware: {hardware}")


def architecture(hardware):
    return {"a100": "80", "rtxpro6000": "120"}[hardware]


def rated_roof(name):
    if re.search(r"A100.*SXM.*80GB", name):
        peak, bandwidth = 19.5, 2039
        source = "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf"
    elif matches(name, "rtxpro6000"):
        base = "https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/"
        if "Max-Q" in name:
            peak, bandwidth, source = 110, 1792, base + "rtx-pro-6000-max-q/"
        elif "Server" in name:
            peak, bandwidth = 120, 1597
            source = "https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/"
        elif "Workstation" in name:
            peak, bandwidth, source = 125, 1792, base + "rtx-pro-6000/"
        else:
            raise ValueError(f"GPU edition missing from saved name; cannot choose rated ceilings: {name}")
    else:
        raise ValueError(f"No verified rated scalar roofline for {name}")
    return {"hardware": name, "peak_fp32_flops_per_s": peak * 1e12,
            "peak_dram_bytes_per_s": bandwidth * 1e9, "roof_source": source,
            "ceilings_are_per_gpu": True, "specification_checked": "2026-09-14"}
