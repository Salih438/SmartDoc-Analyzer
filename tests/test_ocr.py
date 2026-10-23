from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ocr_document import OCR_GROUND_TRUTH, make_ocr_document
from src.ocr.engine import OcrResult, _ocr_image, available_languages, check_installation, correct_orientation, is_available, run_ocr


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


def test_missing_tesseract_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ocr.engine as engine

    monkeypatch.setattr(engine.shutil, "which", lambda _: None)
    engine.available_languages.cache_clear()
    with pytest.raises(RuntimeError, match="Tesseract bulunamadı"):
        run_ocr(np.full((10, 10), 255, dtype=np.uint8))


def test_orientation_uses_osd_rotation_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ocr.engine as engine

    class FakeTesseract:
        @staticmethod
        def image_to_osd(image: np.ndarray, config: str) -> str:
            return "Rotate: 90\nOrientation in degrees: 270"

    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    monkeypatch.setattr(engine, "_pytesseract", lambda: FakeTesseract())
    corrected, warning = correct_orientation(image)
    assert np.array_equal(corrected, np.rot90(image, k=3))
    assert warning is None


def test_orientation_falls_back_to_four_confidence_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ocr.engine as engine

    class FakeTesseract:
        @staticmethod
        def image_to_osd(image: np.ndarray, config: str) -> str:
            raise RuntimeError("osd data unavailable")

    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    trial_count = 0

    used_languages: list[str] = []

    def fake_ocr(candidate: np.ndarray, lang: str, psm: int) -> OcrResult:
        nonlocal trial_count
        trial_count += 1
        used_languages.append(lang)
        confidence = 90.0 if np.array_equal(candidate, np.rot90(image, k=2)) else 10.0
        return OcrResult("sample", confidence, [], 0.0)

    monkeypatch.setattr(engine, "_pytesseract", lambda: FakeTesseract())
    monkeypatch.setattr(engine, "run_ocr", fake_ocr)
    corrected, warning = correct_orientation(image, lang="tur")
    assert trial_count == 4
    assert used_languages == ["tur"] * 4
    assert np.array_equal(corrected, np.rot90(image, k=2))
    assert warning is not None


def test_available_languages_is_cached_and_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ocr.engine as engine

    class FakeTesseract:
        calls = 0

        @classmethod
        def get_languages(cls, config: str) -> list[str]:
            cls.calls += 1
            return ["eng", "tur"]

    monkeypatch.setattr(engine, "_pytesseract", lambda: FakeTesseract())
    engine.available_languages.cache_clear()
    assert engine.available_languages() == ("eng", "tur")
    assert engine.available_languages() == ("eng", "tur")
    assert FakeTesseract.calls == 1

    engine.available_languages.cache_clear()
    assert engine.available_languages() == ("eng", "tur")
    assert FakeTesseract.calls == 2
    engine.available_languages.cache_clear()


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
