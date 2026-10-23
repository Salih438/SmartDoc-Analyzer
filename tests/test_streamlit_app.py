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
    assert not app.tabs
    assert app.segmented_control[0].value == "Tara · Scan"
    app.file_uploader[0].set_value(("örnek.png", _encoded_image(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert app.get("download_button") or any(
        button.key == "scan_allow_unrectified" for button in app.button
    )


def _show_advanced(app):
    return next(widget for widget in app.segmented_control if widget.key == "main_view").set_value("Gelişmiş · Advanced").run(timeout=30)


def test_not_found_warns_and_pipeline_does_not_emit_output_without_opt_in() -> None:
    app = _app_test().run()
    uniform = np.full((160, 200, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", uniform)
    assert ok
    app.file_uploader[0].set_value(("uniform.png", encoded.tobytes(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert any("Belge sınırı bulunamadı" in item.value for item in app.warning)
    assert any(button.label == "Bu sayfada tüm görüntüyü kullan · Use full image on this page" for button in app.button)
    assert not app.get("download_button")
    next(button for button in app.button if button.label == "Bu sayfada tüm görüntüyü kullan · Use full image on this page").click().run(timeout=30)
    assert not app.exception
    assert any(button.key == "scan_pdf_download" for button in app.get("download_button"))


def test_pipeline_results_render_in_processing_and_export_tabs_without_duplicate_ids() -> None:
    from src.evaluation.scenes import make_scene

    document = np.full((500, 360, 3), 245, dtype=np.uint8)
    cv2.putText(document, "SMARTDOC", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
    image, _ = make_scene(document, "frontal_easy", seed=12)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    app = _app_test().run()
    _show_advanced(app)
    next(widget for widget in app.selectbox if widget.label == "Siyah-beyaz çıktı · Black and white output").select("otsu").run()
    app.file_uploader[0].set_value(("belge.png", encoded.tobytes(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    downloads = app.get("download_button")
    assert {button.key for button in downloads} >= {
        "processing_result_png",
        "processing_result_json",
        "export_result_png",
        "export_result_json",
    }


def test_comparison_tab_renders_enhancement_section() -> None:
    app = _app_test().run()
    _show_advanced(app)
    app.file_uploader[0].set_value(("örnek.png", _encoded_image(), "image/png"))
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Sonuçları karşılaştır · Compare results").click().run(timeout=30)

    assert not app.exception
    assert any(item.value == "Aydınlatma iyileştirme" for item in app.subheader)


def test_corrupt_upload_is_shown_as_streamlit_error_not_exception() -> None:
    app = _app_test().run()
    app.file_uploader[0].set_value(("bozuk.png", b"broken image", "image/png"))
    app.run(timeout=20)
    assert not app.exception
    assert app.error
    assert "Desteklenmeyen" in app.error[0].value


def test_ocr_tab_reports_missing_tesseract_without_crashing() -> None:
    from src.ocr.engine import is_available

    app = _app_test().run()
    _show_advanced(app)
    app.file_uploader[0].set_value(("örnek.png", _encoded_image(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    if not is_available():
        assert any("Metin tanıma kullanılamıyor" in item.value for item in app.info)


def test_scan_tab_pipeline_scan_and_pdf_download_button() -> None:
    from src.evaluation.scenes import make_scene

    document = np.full((500, 360, 3), 245, dtype=np.uint8)
    cv2.putText(document, "SCAN TEST", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
    image, _ = make_scene(document, "frontal_easy", seed=32)
    success, encoded = cv2.imencode(".png", image)
    assert success

    app = _app_test().run()
    app.file_uploader[0].set_value(("belge.png", encoded.tobytes(), "image/png"))
    app.run(timeout=20)
    assert not app.exception
    if not any(button.key == "scan_pdf_download" for button in app.get("download_button")):
        next(button for button in app.button if button.key.startswith("scan_approve_")).click().run(timeout=30)
    assert any(button.label == "PNG indir · Download PNG" for button in app.get("download_button"))
    pdf_button = next(
        button for button in app.get("download_button") if button.label == "PDF indir (A4) · Download PDF"
    )
    assert pdf_button.proto.url.startswith("data:application/pdf") or pdf_button.proto.url

    next(widget for widget in app.segmented_control if widget.key == "scan_mode").set_value("Siyah-Beyaz · B&W").run(timeout=30)
    assert not app.exception
    assert {button.key for button in app.get("download_button")} >= {
        "scan_png_download", "scan_pdf_download"
    }
    assert app.session_state["scan_result_key"][1] == "bw"


def test_scan_uploads_multiple_pages_into_one_a4_pdf() -> None:
    from src.evaluation.scenes import make_scene

    uploads = []
    for seed in (41, 42):
        document = np.full((500, 360, 3), 245, dtype=np.uint8)
        cv2.putText(document, f"PAGE {seed}", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
        image, _ = make_scene(document, "frontal_easy", seed=seed)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        uploads.append((f"page-{seed}.png", encoded.tobytes(), "image/png"))
    uploads.insert(0, ("broken.png", b"not an image", "image/png"))

    app = _app_test().run()
    app.file_uploader[0].set_value(uploads)
    app.run(timeout=60)

    assert not app.exception
    assert app.error
    assert any(button.key == "scan_pdf_download" for button in app.get("download_button"))
    pdf_button = next(button for button in app.get("download_button") if button.key == "scan_pdf_download")
    assert pdf_button.proto.url.endswith(".pdf")


def test_scan_auto_mode_change_keeps_result_and_pdf_download() -> None:
    from src.evaluation.scenes import make_scene

    doc = np.full((460, 340, 3), 245, dtype=np.uint8)
    cv2.putText(doc, "AUTO SCAN", (28, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3, cv2.LINE_AA)
    scene, _ = make_scene(doc, "frontal_easy", seed=52)
    ok, encoded = cv2.imencode(".png", scene)
    assert ok
    app = _app_test().run()
    app.file_uploader[0].set_value(("auto.png", encoded.tobytes(), "image/png"))
    app.run(timeout=30)
    if not any(button.key == "scan_pdf_download" for button in app.get("download_button")):
        next(button for button in app.button if button.key.startswith("scan_approve_")).click().run(timeout=30)
    assert not app.exception
    next(widget for widget in app.segmented_control if widget.key == "scan_mode").set_value("Gri · Gray").run(timeout=30)
    assert not app.exception
    assert app.session_state["scan_result_key"][1] == "gray"
    assert any(button.key == "scan_pdf_download" for button in app.get("download_button"))


def test_advanced_controls_are_not_rendered_on_scan_and_reset_to_defaults() -> None:
    app = _app_test().run()
    assert not app.exception
    assert not app.selectbox
    _show_advanced(app)
    assert not app.exception
    assert any(widget.key == "cfg_detect" for widget in app.checkbox)
    detect_denoise = next(widget for widget in app.checkbox if widget.key == "cfg_detect_denoise")
    detect_denoise.check().run()
    assert app.session_state["cfg_detect_denoise"] is True
    next(button for button in app.button if button.key == "cfg_reset").click().run()
    assert not app.exception
    assert app.session_state["cfg_detect_denoise"] is False


def test_natural_filename_sorting_orders_page2_before_page10() -> None:
    from app.main import _natural_sort_key

    assert sorted(["sayfa10.jpg", "sayfa2.jpg"], key=_natural_sort_key) == [
        "sayfa2.jpg", "sayfa10.jpg"
    ]


def test_scan_page_selection_and_reorder_state_follow_natural_order() -> None:
    app = _app_test().run()
    payload = _encoded_image()
    app.file_uploader[0].set_value([
        ("sayfa10.png", payload, "image/png"),
        ("sayfa2.png", payload, "image/png"),
    ])
    app.run(timeout=30)
    assert not app.exception
    assert len(app.session_state["scan_page_order"]) == 2
    buttons = [button for button in app.button if button.key.startswith("scan_select_")]
    buttons[1].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["scan_selected_index"] == 1
    before = list(app.session_state["scan_page_order"])
    next(button for button in app.button if button.key.startswith("scan_up_") and not button.disabled).click().run(timeout=30)
    assert app.session_state["scan_page_order"] == [before[1], before[0]]
    next(button for button in app.button if button.key.startswith("scan_remove_")).click().run(timeout=30)
    assert len(app.session_state["scan_page_order"]) == 1


def test_detection_overlay_only_uses_status_appropriate_edges() -> None:
    from app.main import _detection_overlay
    from src.document.detection import DetectionResult

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    corners = np.array([[10, 10], [80, 10], [80, 60], [10, 60]], dtype=np.float32)
    for status, expected_color in (("ok", (0, 180, 90)), ("low_confidence", (0, 180, 255))):
        detection = DetectionResult(corners, status, 0.5, "test", "test")
        overlay = _detection_overlay(image, detection)
        assert tuple(overlay[10, 30]) == expected_color
    missing = DetectionResult(None, "not_found", 0.0, "test", "test")
    np.testing.assert_array_equal(_detection_overlay(image, missing), image)
