"""Sentetik tarama örnekleri üretim scriptinin testi."""

from __future__ import annotations

from pathlib import Path

import cv2

from scripts.make_scan_samples import generate_samples


def test_generate_scan_samples_writes_all_scene_mode_pngs(tmp_path: Path) -> None:
    generated = generate_samples(tmp_path)

    assert len(generated) == 12
    assert all(path.is_file() and path.suffix == ".png" for path in generated)
    assert all(cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None for path in generated)
