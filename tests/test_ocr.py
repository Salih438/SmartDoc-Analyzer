from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ocr_document import OCR_GROUND_TRUTH, make_ocr_document
from src.ocr.engine import _ocr_image, available_languages, check_installation, is_available


def test_ocr_input_polarity_converts_white_foreground_to_black_ink() -> None:
    white_ink = np.zeros((20, 30), dtype=np.uint8)
    white_ink[5:15, 8:20] = 255
    prepared = _ocr_image(white_ink)
    assert prepared[10, 10] == 0
    assert prepared[0, 0] == 255


def test_ocr_input_keeps_black_ink_white_paper() -> None:
    black_ink = np.full((20, 30), 255, dtype=np.uint8)
    black_ink[5:15, 8:20] = 0
    assert np.array_equal(_ocr_image(black_ink), black_ink)


def test_ocr_input_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ocr_image(np.zeros((10, 10), dtype=np.float32))


def test_turkish_ocr_document_uses_ttf_and_known_text() -> None:
    image, mask, text = make_ocr_document(700, 900)
    assert image.shape == (900, 700, 3)
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})
    assert text == OCR_GROUND_TRUTH
    assert all(letter in text for letter in "çşğıüöİ")


@pytest.mark.skipif(not is_available(), reason="Tesseract ikilisi kurulu değil")
def test_tesseract_installation_reports_version_and_requested_languages() -> None:
    status = check_installation()
    assert status["version"]
    languages = set(available_languages())
    assert isinstance(languages, set)
    assert set(status["missing"]).issubset({"tur", "eng", "osd"})
