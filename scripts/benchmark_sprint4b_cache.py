"""Sekiz 12 MP sayfada SmartDoc önbellek çağrılarını ölçer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import gc

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as ui


def make_upload(seed: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    height, width = 3000, 4000
    yy, xx = np.mgrid[:height, :width]
    base = 112 + 14 * xx / width + 8 * yy / height
    noise = rng.normal(0, 3.5, (height, width))
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(gray, (400, 250), (3600, 2750), 246, -1)
    cv2.rectangle(gray, (400, 250), (3600, 2750), 35, 7)
    for row in range(650, 2300, 170):
        cv2.line(gray, (650, row), (3350, row), 70, 3)
        cv2.putText(gray, f"PAGE {seed} DOCUMENT TEXT", (700, row - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 20, 3, cv2.LINE_AA)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), hashlib.sha256(gray.tobytes()).hexdigest()


def instrument(name: str) -> dict[str, int]:
    cached = getattr(ui, name)
    counter = {"calls": 0, "misses": 0}
    original = cached._info.func

    def counted(*args, **kwargs):
        counter["misses"] += 1
        return original(*args, **kwargs)

    cached.clear()
    cached._info.func = counted
    return counter


def main() -> None:
    counters = {
        key: instrument(key)
        for key in ("_cached_pipeline", "_cached_scan", "_cached_flattened_scan_source", "_cached_pdf")
    }
    pages = [make_upload(seed) for seed in range(800, 808)]
    config_json = json.dumps(ui.asdict(ui.TARA_CONFIG), sort_keys=True)
    total_calls = {"_cached_pipeline": 40, "_cached_scan": 45, "_cached_flattened_scan_source": 32, "_cached_pdf": 5}

    def run_sequence(label: str) -> None:
        for counter in counters.values():
            counter["misses"] = 0
        for name in counters:
            getattr(ui, name).clear()
        processed = []
        start = time.perf_counter()
        for mode in ("color", "gray", "bw", "original", "color"):
            processed = [
                (image, image_key, ui._cached_pipeline(image, image_key, config_json, retain_intermediates=False))
                for image, image_key in pages
            ]
            selected_source = processed[0][2].rectified_image
            if selected_source is None:
                selected_source = processed[0][0]
            ui._cached_scan(selected_source, pages[0][1], mode)
            scans = [
                ui._cached_scan(result.rectified_image if result.rectified_image is not None else image, image_key, mode)
                for image, image_key, result in processed
            ]
            batch_key = hashlib.sha256((mode + "".join(key for _, key in pages)).encode("ascii")).hexdigest()
            ui._cached_pdf(scans, batch_key, mode, "a4", "auto")
        elapsed = time.perf_counter() - start
        print(label, "elapsed_seconds", round(elapsed, 3))
        for name, counter in counters.items():
            calls = total_calls[name]
            print(label, name, "calls", calls, "misses", counter["misses"], "hits", calls - counter["misses"])
        del processed
        gc.collect()

    for name in counters:
        getattr(ui, name)._info.max_entries = 6
    run_sequence("before_max_entries_6")
    for name in counters:
        getattr(ui, name)._info.max_entries = {
            "_cached_pipeline": 8,
            "_cached_scan": 32,
            "_cached_flattened_scan_source": 8,
            "_cached_pdf": 8,
        }[name]
    run_sequence("after_page_capacity")


if __name__ == "__main__":
    main()
