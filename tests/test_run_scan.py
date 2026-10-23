"""Tek görüntü/klasör tarama CLI akışının testleri."""

from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
import pytest

from scripts.run_scan import _image_files, run_scan
from src.analysis.image_info import save_image


def _scene(seed: int) -> np.ndarray:
    from src.evaluation.scenes import make_scene

    document = np.full((500, 360, 3), 245, dtype=np.uint8)
    cv2.putText(document, "CLI SCAN", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
    return make_scene(document, "frontal_easy", seed)[0]


def test_folder_images_are_sorted_case_insensitively(tmp_path: Path) -> None:
    save_image(tmp_path / "b.png", _scene(2))
    save_image(tmp_path / "A.png", _scene(1))

    assert [path.name for path in _image_files(tmp_path)] == ["A.png", "b.png"]


def test_run_scan_creates_multipage_pdf_for_unicode_folder(tmp_path: Path) -> None:
    folder = tmp_path / "Çalışma Belgeleri"
    folder.mkdir()
    save_image(folder / "belge_b.png", _scene(2))
    save_image(folder / "belge_a.png", _scene(1))
    output = tmp_path / "çıktı.pdf"

    count = run_scan(folder, output, mode="gray", page_size="fit", dpi=120)
    contents = output.read_bytes()

    assert count == 2
    assert contents.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", contents)) == 2


def test_scan_rejects_empty_folder_and_does_not_create_pdf(tmp_path: Path) -> None:
    output = tmp_path / "empty.pdf"
    with pytest.raises(ValueError, match="fotoğraf bulunamadı"):
        run_scan(tmp_path, output)
    assert not output.exists()
