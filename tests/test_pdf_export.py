"""PDF dışa aktarma, sayfa boyutu ve girdi doğrulama testleri."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pytest
from PIL import Image, features

from src.export.pdf import _a4_dimensions_px, _page_canvas, _to_pdf_image, images_to_pdf


def _page(value: int = 235, shape: tuple[int, int] = (180, 240)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_single_gray_page_pdf_starts_with_pdf_signature() -> None:
    output = BytesIO()
    images_to_pdf([_page()], output, page_size="fit", dpi=150)

    assert output.getvalue().startswith(b"%PDF")


def test_two_bgr_pages_are_written_to_one_pdf() -> None:
    output = BytesIO()
    pages = [np.full((80, 100, 3), (20, 70, 140), dtype=np.uint8) for _ in range(2)]
    images_to_pdf(pages, output, page_size="fit", dpi=150)
    contents = output.getvalue()

    assert contents.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", contents)) == 2


@pytest.mark.parametrize(
    ("image", "expected_filter"),
    [
        (_page(), b"/DCTDecode"),
        (
            np.where(_page() > 200, 255, 0).astype(np.uint8),
            b"/CCITTFaxDecode" if features.check("libtiff") else b"/DCTDecode",
        ),
        (np.repeat(_page()[:, :, None], 3, axis=2), b"/DCTDecode"),
    ],
    ids=("grayscale-quality-jpeg", "binary-lossless-when-available", "color-quality-jpeg"),
)
def test_pdf_embeds_pages_with_expected_filter(image: np.ndarray, expected_filter: bytes) -> None:
    output = BytesIO()
    images_to_pdf([image], output, page_size="fit", dpi=150)
    contents = output.getvalue()

    assert contents.startswith(b"%PDF")
    assert expected_filter in contents


def test_mixed_gray_and_bw_pages_have_correct_pdf_page_count() -> None:
    output = BytesIO()
    images_to_pdf([_page(), np.where(_page() > 200, 255, 0).astype(np.uint8)], output)

    assert len(re.findall(rb"/Type\s*/Page\b", output.getvalue())) == 2


def test_a4_300_dpi_canvas_dimensions_are_exactly_expected() -> None:
    assert _a4_dimensions_px(300) == (2480, 3508)


def test_a4_auto_orientation_uses_landscape_for_wide_images() -> None:
    page = _to_pdf_image(np.full((120, 240), 255, np.uint8), 1)
    canvas = _page_canvas(page, "a4", 300, 0, "auto")

    assert canvas.size == (3508, 2480)


def test_a4_explicit_portrait_overrides_auto_orientation() -> None:
    page = _to_pdf_image(np.full((120, 240), 255, np.uint8), 1)
    canvas = _page_canvas(page, "a4", 300, 0, "portrait")

    assert canvas.size == (2480, 3508)


def test_bw_resize_keeps_ccitt_and_preserves_thin_lines_better_than_nearest() -> None:
    image = np.full((4000, 3000), 255, np.uint8)
    line_x = list(range(120, 2880, 90))
    for x in line_x:
        image[300:3700, x] = 0
    page = _to_pdf_image(image, 1, lossless_binary=True)
    actual = np.asarray(_page_canvas(page, "a4", 300, 0, "portrait").convert("L"))
    nearest = np.asarray(
        page.resize((2480, 3307), Image.Resampling.NEAREST)
        .convert("L")
    )
    actual_positions = [round(x * 2480 / 3000) for x in line_x]
    nearest_positions = [round(x * 2480 / 3000) for x in line_x]

    def retention(array: np.ndarray, positions: list[int], y_offset: int) -> float:
        preserved = sum(
            bool(np.any(array[y_offset + 200 : y_offset + 3100, x : x + 1] < 128))
            for x in positions
        )
        return preserved / len(positions)

    assert retention(actual, actual_positions, 100) > retention(nearest, nearest_positions, 0)
    assert _page_canvas(page, "a4", 300, 0, "portrait").mode == "1"


def test_pdf_can_be_written_to_unicode_path(tmp_path: Path) -> None:
    target = tmp_path / "Çalışma Örneği" / "çıktı.pdf"
    images_to_pdf([_page()], target, page_size="a4", dpi=72)

    assert target.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("images", "output", "kwargs", "error"),
    [
        ([], BytesIO(), {}, "en az bir görüntü"),
        ([_page().astype(np.float32)], BytesIO(), {}, "uint8"),
        ([np.zeros((20, 30, 4), dtype=np.uint8)], BytesIO(), {}, "BGR"),
        ([_page()], BytesIO(), {"page_size": "letter"}, "page_size"),
        ([_page()], BytesIO(), {"dpi": 0}, "dpi"),
        ([_page()], BytesIO(), {"margin_mm": -1}, "margin_mm"),
    ],
)
def test_pdf_rejects_invalid_inputs(images, output, kwargs, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        images_to_pdf(images, output, **kwargs)
