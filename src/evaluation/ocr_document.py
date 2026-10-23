"""Türkçe karakter içeren, TrueType fontla çizilmiş OCR test belgesi."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OCR_GROUND_TRUTH = "SmartDoc Analyzer ÇAĞRI İSTASYONU çağrı Çalışma ölçümü: ışık, gölge ve görüntü."


def _font_path() -> Path:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Türkçe OCR sentetiği için Türkçe glifli TrueType font bulunamadı.")


def make_ocr_document(width: int = 1000, height: int = 1400) -> tuple[np.ndarray, np.ndarray, str]:
    """Return BGR page, white-text GT mask, and exact rendered text."""
    if width < 400 or height < 500:
        raise ValueError("OCR belgesi için en az 400x500 boyut gerekir.")
    font = ImageFont.truetype(str(_font_path()), size=max(24, round(width * 0.032)))
    page = Image.new("RGB", (width, height), "white")
    mask = Image.new("L", (width, height), 0)
    draw, mask_draw = ImageDraw.Draw(page), ImageDraw.Draw(mask)
    lines = ["SmartDoc Analyzer", "ÇAĞRI İSTASYONU", "çağrı", "Çalışma ölçümü:", "ışık, gölge ve görüntü."]
    y = round(height * 0.16)
    for line in lines:
        draw.text((round(width * 0.1), y), line, font=font, fill="black")
        mask_draw.text((round(width * 0.1), y), line, font=font, fill=255)
        y += round(height * 0.09)
    rgb = np.asarray(page, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask_array = np.asarray(mask, dtype=np.uint8)
    binary_mask = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    return bgr, binary_mask, OCR_GROUND_TRUTH
