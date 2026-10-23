"""Tesseract OCR bağdaştırıcısı; Tesseract sistem ikilisi gerektirir."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import time
from typing import Any

import cv2
import numpy as np


@dataclass
class OcrWord:
    """Bir OCR sözcüğü, güven puanı ve (x, y, w, h) kutusu."""

    text: str
    conf: float
    bbox: tuple[int, int, int, int]


@dataclass
class OcrResult:
    """OCR metni, sözcük sonuçları, ortalama güven ve geçen süre."""

    text: str
    mean_conf: float
    words: list[OcrWord]
    duration_ms: float


def _pytesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR için pytesseract paketi kurulu değil.") from exc
    if shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract bulunamadı. Tesseract'ı kurup PATH'e ekleyin.")
    return pytesseract


def is_available() -> bool:
    """True if the executable and pytesseract binding are both available."""
    try:
        pytesseract = _pytesseract()
        pytesseract.get_tesseract_version()
        return True
    except (RuntimeError, OSError):
        return False


def available_languages() -> tuple[str, ...]:
    """Return installed Tesseract language codes, or raise a Turkish error."""
    pytesseract = _pytesseract()
    try:
        return tuple(pytesseract.get_languages(config=""))
    except Exception as exc:  # Tesseract raises several platform-specific errors.
        raise RuntimeError(f"Tesseract dil verileri okunamadı: {exc}") from exc


def check_installation() -> dict[str, object]:
    """Inspect executable, version and tur/eng/osd language availability."""
    executable = shutil.which("tesseract")
    result: dict[str, object] = {"executable": executable, "version": None, "languages": (), "missing": []}
    try:
        pytesseract = _pytesseract()
        result["version"] = str(pytesseract.get_tesseract_version())
        languages = available_languages()
        result["languages"] = languages
        result["missing"] = [name for name in ("tur", "eng", "osd") if name not in languages]
    except RuntimeError as exc:
        result["error"] = str(exc)
    return result


def _ocr_image(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise TypeError("OCR uint8 görüntü bekliyor.")
    if img.ndim == 3:
        if img.shape[2] != 3:
            raise ValueError("OCR gri veya BGR görüntü bekliyor.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img.copy()
    else:
        raise ValueError("OCR gri veya BGR görüntü bekliyor.")
    # Pipeline foreground is 255 (white ink on black); Tesseract expects black ink.
    if float(np.mean(gray)) < 127.5:
        gray = cv2.bitwise_not(gray)
    return gray


def run_ocr(img: np.ndarray, lang: str = "tur+eng", psm: int = 6) -> OcrResult:
    """Run Tesseract on black text/white paper and return word boxes/confidence."""
    pytesseract = _pytesseract()
    installed = set(available_languages())
    requested = [language for language in lang.split("+") if language]
    missing = [language for language in requested if language not in installed]
    if missing:
        raise RuntimeError(f"Tesseract dil verisi eksik: {', '.join(missing)}")
    prepared = _ocr_image(img)
    started = time.perf_counter()
    try:
        data = pytesseract.image_to_data(
            prepared, lang=lang, config=f"--psm {int(psm)}", output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        raise RuntimeError(f"OCR çalıştırılamadı: {exc}") from exc
    words: list[OcrWord] = []
    for i, value in enumerate(data.get("text", [])):
        text = str(value).strip()
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError, IndexError):
            confidence = -1.0
        if text and confidence >= 0:
            words.append(OcrWord(text, confidence, (
                int(data["left"][i]), int(data["top"][i]),
                int(data["width"][i]), int(data["height"][i]),
            )))
    text = " ".join(word.text for word in words)
    mean_conf = float(np.mean([word.conf for word in words])) if words else 0.0
    return OcrResult(text, mean_conf, words, (time.perf_counter() - started) * 1000.0)


def correct_orientation(img: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Use OSD first; if it cannot decide, choose rotation with highest OCR confidence."""
    pytesseract = _pytesseract()
    prepared = _ocr_image(img)
    warning: str | None = None
    try:
        osd = pytesseract.image_to_osd(prepared, config="--psm 0")
        rotation = int(next(line.split(":", 1)[1] for line in osd.splitlines() if line.startswith("Rotate:")))
        if rotation in {0, 90, 180, 270}:
            return np.rot90(img, k=-(rotation // 90)).copy(), None
        warning = "OSD geçerli bir yön döndürmedi; OCR güveniyle yön aranıyor."
    except Exception:
        warning = "OSD yönü belirleyemedi; OCR güveniyle 0/90/180/270 deneniyor."
    rotations = [np.rot90(img, k=k).copy() for k in (0, 3, 2, 1)]
    scored: list[tuple[float, np.ndarray]] = []
    for candidate in rotations:
        try:
            scored.append((run_ocr(candidate, lang="eng", psm=6).mean_conf, candidate))
        except RuntimeError:
            scored.append((0.0, candidate))
    return max(scored, key=lambda item: item[0])[1], warning
