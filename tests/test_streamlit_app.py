from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.analysis.image_info import load_image_from_bytes


def _encoded_image(extension: str = ".png") -> bytes:
    image = np.full((180, 220, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (200, 160), (20, 20, 20), 2)
    cv2.putText(image, "TEST", (55, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_load_image_from_supported_bytes(extension: str) -> None:
    decoded = load_image_from_bytes(_encoded_image(extension))
    assert decoded.shape == (180, 220, 3)
    assert decoded.dtype == np.uint8


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_load_image_from_bytes_rejects_empty_unsupported_and_corrupt(payload: bytes) -> None:
    with pytest.raises(ValueError, match="Görüntü|Desteklenmeyen"):
        load_image_from_bytes(payload)


def test_preview_is_scaled_copy_and_keeps_input_unchanged() -> None:
    from app.components.image_viewer import preview_image

    original = np.full((900, 700, 3), 30, dtype=np.uint8)
    saved = original.copy()
    preview = preview_image(original, max_dimension=350)
    assert max(preview.shape[:2]) == 350
    assert np.array_equal(original, saved)


def test_download_stem_keeps_unicode_and_removes_path_separators() -> None:
    from app.main import _safe_download_stem

    assert _safe_download_stem("şube/Çalışma Örneği.png") == "Çalışma Örneği"
    assert _safe_download_stem("../?.jpg") == "_"


def _app_test():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    return streamlit_testing.AppTest.from_file(str(app_path))


def test_streamlit_app_launches_and_uploaded_image_renders_tabs() -> None:
    app = _app_test().run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.file_uploader[0].set_value(("örnek.png", _encoded_image(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert any(element.value == "Belge sınırı tespiti" for element in app.subheader)
    assert len(app.tabs) == 5


def test_not_found_warns_and_pipeline_does_not_emit_output_without_opt_in() -> None:
    app = _app_test().run()
    uniform = np.full((160, 200, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", uniform)
    assert ok
    app.file_uploader[0].set_value(("uniform.png", encoded.tobytes(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert any("güvenilir biçimde tespit edilemedi" in item.value.lower() for item in app.warning)
    next(button for button in app.button if button.label == "Pipeline çalıştır").click().run(timeout=20)
    assert not app.exception
    assert any("onay kapalıysa" in item.value.lower() for item in app.info)
    assert not app.get("download_button")


def test_corrupt_upload_is_shown_as_streamlit_error_not_exception() -> None:
    app = _app_test().run()
    app.file_uploader[0].set_value(("bozuk.png", b"broken image", "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert app.error
    assert "Desteklenmeyen" in app.error[0].value
