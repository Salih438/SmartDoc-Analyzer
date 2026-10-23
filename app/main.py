"""SmartDoc uygulama arayüzü; görüntü işleme algoritmaları ``src`` içindedir."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from typing import Any, get_args, get_type_hints

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.comparison_view import render_comparisons
from app.components.image_viewer import display_image
from app.i18n import t
from app.styles import inject_styles
from src.analysis.histogram import plot_histogram
from src.analysis.image_info import get_image_info, load_image_from_bytes
from src.document.detection import DetectionResult
from src.evaluation.metrics import sharpness_laplacian_var
from src.export.pdf import images_to_pdf
from src.ocr.engine import check_installation, correct_orientation, is_available, run_ocr
from src.pipeline import PipelineConfig, PipelineResult, process_image, select_ocr_input
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.illumination import flatten_background
from src.scan import SCAN_MODES, make_scan


TARA_CONFIG = PipelineConfig(
    detect=True,
    detection_fallback=True,
    enhance="none",
    denoise="none",
    sharpen="none",
    binarize="none",
    morphology="none",
)
MAX_CACHE_ENTRIES = 8
MAX_SCAN_CACHE_ENTRIES = 32
MAX_PDF_CACHE_ENTRIES = 8
MAX_UPLOAD_PAGES = 8
SCAN_LABELS = {
    "color": "mode_color",
    "gray": "mode_gray",
    "bw": "mode_bw",
    "original": "mode_original",
}
CONFIG_WIDGETS = {
    "cfg_detect", "cfg_strategy", "cfg_canny", "cfg_detect_denoise",
    "cfg_detection_fallback", "cfg_enhance", "cfg_denoise", "cfg_sharpen",
    "cfg_binarize", "cfg_morphology", "cfg_allow_unrectified",
}


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def _decode_uploaded_image(data: bytes) -> np.ndarray:
    """Safely cache image decoding by the uploaded file bytes."""
    return load_image_from_bytes(data)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def _cached_image_analysis(_image: np.ndarray, image_key: str) -> dict[str, object]:
    """Cache pure image analysis by explicit content identity."""
    del image_key
    information: dict[str, object] = get_image_info(_image)
    information["sharpness"] = sharpness_laplacian_var(to_grayscale(_image))
    return information


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def _cached_pipeline(
    _image: np.ndarray, image_key: str, config_json: str, retain_intermediates: bool = True
) -> PipelineResult:
    """Cache processing by hash/config; retain diagnostics only for Advanced."""
    del image_key
    config = PipelineConfig(**json.loads(config_json))
    result = process_image(_image.copy(), config)
    if retain_intermediates:
        return result
    compact_detection = replace(result.detection, debug={})
    return replace(
        result,
        original=_image,
        gray=None,
        enhanced=None,
        denoised=None,
        binary=None,
        detection=compact_detection,
        steps=[],
        info_before={},
        info_after={},
    )


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def _cached_flattened_scan_source(
    _image: np.ndarray, image_key: str
) -> np.ndarray:
    """Cache the costly background estimate using a separate content hash."""
    del image_key
    return flatten_background(_image)


@st.cache_data(show_spinner=False, max_entries=MAX_SCAN_CACHE_ENTRIES)
def _cached_scan(_image: np.ndarray, image_key: str, mode: str) -> np.ndarray:
    """Cache inexpensive scan rendering by source identity and selected mode."""
    if mode == "original":
        return make_scan(_image, mode)  # type: ignore[arg-type]
    flattened = _cached_flattened_scan_source(_image, image_key)
    return make_scan(flattened, mode, background_flattened=True)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False, max_entries=MAX_PDF_CACHE_ENTRIES)
def _cached_pdf(
    _images: list[np.ndarray], batch_key: str, mode: str,
    page_size: str = "a4", a4_orientation: str = "auto",
) -> bytes:
    """PDF'yi sıralı sayfa kimliği, mod ve sayfa ayarlarıyla önbelleğe alır."""
    del batch_key, mode
    output = BytesIO()
    images_to_pdf(
        _images, output, page_size=page_size, dpi=300,
        a4_orientation=a4_orientation,
    )
    return output.getvalue()


def _literal_options(field: str) -> tuple[str, ...]:
    """Read valid options from PipelineConfig Literal annotations."""
    choices = get_args(get_type_hints(PipelineConfig)[field])
    if not choices or not all(isinstance(choice, str) for choice in choices):
        raise TypeError(f"PipelineConfig.{field} alanında seçenek tanımı bulunamadı.")
    return tuple(choices)


def _option_label(field: str, value: str) -> str:
    """Translate internal option values into concise user-facing labels."""
    key = f"{field}_{value}"
    try:
        return t(key)
    except KeyError:
        return value


def _safe_download_stem(filename: str) -> str:
    """Keep Unicode basename characters while removing unsafe name characters."""
    cleaned = re.sub(r"[\\/<>:\"|?*\x00-\x1f]", "_", Path(filename).stem).strip(" .")
    return cleaned or "smartdoc"


def _make_config() -> PipelineConfig:
    """Build the Advanced-only configuration from its sidebar controls."""
    with st.sidebar:
        st.caption(t("advanced_only"))
        if st.button(t("reset"), key="cfg_reset"):
            for widget_key in CONFIG_WIDGETS:
                st.session_state.pop(widget_key, None)
            st.rerun()
        st.subheader(t("settings"))
        defaults = PipelineConfig()
        detect = st.checkbox(t("detect"), value=defaults.detect, key="cfg_detect")
        strategy = st.selectbox(
            t("strategy"), _literal_options("detection_strategy"),
            index=_literal_options("detection_strategy").index(defaults.detection_strategy),
            key="cfg_strategy", format_func=lambda value: _option_label("detection_strategy", value),
        )
        canny_mode = st.selectbox(
            t("canny"), _literal_options("canny_mode"),
            index=_literal_options("canny_mode").index(defaults.canny_mode), key="cfg_canny",
            format_func=lambda value: _option_label("canny_mode", value),
        )
        detect_denoise = st.checkbox(
            t("detect_denoise"), value=defaults.detect_denoise, key="cfg_detect_denoise"
        )
        detection_fallback = st.checkbox(
            t("fallback"), value=defaults.detection_fallback, key="cfg_detection_fallback"
        )
        enhance = st.selectbox(
            t("enhance"), _literal_options("enhance"),
            index=_literal_options("enhance").index(defaults.enhance), key="cfg_enhance",
            format_func=lambda value: _option_label("enhance", value),
        )
        denoise = st.selectbox(
            t("denoise"), _literal_options("denoise"),
            index=_literal_options("denoise").index(defaults.denoise), key="cfg_denoise",
            format_func=lambda value: _option_label("denoise", value),
        )
        sharpen = st.selectbox(
            t("sharpen"), _literal_options("sharpen"),
            index=_literal_options("sharpen").index(defaults.sharpen), key="cfg_sharpen",
            format_func=lambda value: _option_label("sharpen", value),
        )
        binarize = st.selectbox(
            t("binarize"), _literal_options("binarize"),
            index=_literal_options("binarize").index(defaults.binarize), key="cfg_binarize",
            format_func=lambda value: _option_label("binarize", value),
        )
        morphology = st.selectbox(
            t("morphology"), _literal_options("morphology"),
            index=_literal_options("morphology").index(defaults.morphology), key="cfg_morphology",
            format_func=lambda value: _option_label("morphology", value),
        )
        allow_unrectified = st.checkbox(
            t("allow_unrectified"), value=defaults.allow_unrectified,
            key="cfg_allow_unrectified",
        )
    return PipelineConfig(
        detect=detect,
        detection_strategy=strategy,
        canny_mode=canny_mode,
        detect_denoise=detect_denoise,
        detection_fallback=detection_fallback,
        enhance=enhance,
        denoise=denoise,
        sharpen=sharpen,
        binarize=binarize,
        morphology=morphology,
        allow_unrectified=allow_unrectified,
    )


def _detection_overlay(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """Duruma göre belge sınırını kopya görüntü üzerine çizer."""
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    if detection.status in {"ok", "low_confidence"} and detection.corners is not None:
        color = (0, 180, 90) if detection.status == "ok" else (0, 180, 255)
        cv2.polylines(overlay, [np.rint(detection.corners).astype(np.int32)], True, color, 3)
    return overlay


def _natural_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    """Dosya adlarını doğal sıralar; ör. sayfa2, sayfa10'dan önce gelir."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


def _encode_png(image: np.ndarray) -> bytes:
    """Encode one image to PNG bytes."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG çıktısı bellekte oluşturulamadı.")
    return buffer.tobytes()


def _show_error(message: str, exc: Exception, key: str) -> None:
    """Show a concise localized error and keep technical details collapsed."""
    st.error(message)
    with st.expander(t("details"), key=key):
        st.code(str(exc))


def _render_result(
    result: PipelineResult, config: PipelineConfig, source_stem: str, key_prefix: str
) -> None:
    """Render processing output and uniquely keyed downloads."""
    st.write({t("detection"): result.detection.status, t("rectified"): result.rectified})
    for warning in result.warnings:
        st.warning(warning)
    if not result.rectified and result.detection.status != "ok":
        st.info(t("warning_no_rectification"))
        return
    original_column, rectified_column = st.columns(2)
    with original_column:
        display_image(st, result.original, t("original"), max_dimension=1400)
    with rectified_column:
        if result.rectified and result.gray is not None:
            display_image(st, result.gray, t("rectified_gray"))
        elif result.gray is not None:
            display_image(st, result.gray, t("unrectified_gray"))
    for key, image in (("enhanced", result.enhanced), ("denoised", result.denoised), ("binary_output", result.binary)):
        if image is not None:
            display_image(st, image, t(key))
    st.subheader(t("steps"))
    st.dataframe(result.steps, width="stretch")
    st.subheader(t("before_after"))
    st.dataframe(
        [{t("stage"): t("stage_before"), **result.info_before}, {t("stage"): t("stage_after"), **result.info_after}],
        width="stretch",
    )
    if result.binary is not None:
        st.download_button(
            t("advanced_png_download"), data=_encode_png(result.binary),
            file_name=f"{source_stem}_sonuc.png", mime="image/png",
            key=f"{key_prefix}_result_png",
        )
    metadata = {"config": asdict(config), "steps": result.steps, "warnings": result.warnings}
    st.download_button(
        t("advanced_json_download"),
        data=json.dumps(metadata, ensure_ascii=False, indent=2),
        file_name=f"{source_stem}_pipeline.json", mime="application/json",
        key=f"{key_prefix}_result_json",
    )


def _decode_records(uploaded_files: list[Any]) -> list[tuple[Any, bytes, np.ndarray]]:
    """Decode uploaded images and present user-safe errors for invalid files."""
    records = []
    ordered_files = sorted(uploaded_files, key=lambda upload: _natural_sort_key(upload.name))
    for index, item in enumerate(ordered_files):
        payload = item.getvalue()
        try:
            image = _decode_uploaded_image(payload).copy()
            if image.dtype != np.uint8:
                raise TypeError("Desteklenen görüntü türü uint8 olmalıdır.")
            records.append((item, payload, image))
        except (TypeError, ValueError, cv2.error) as exc:
            _show_error(f"{item.name}: {t('unsupported_upload')}", exc, f"upload_err_{index}")
    return records


def _render_empty_advanced(tabs: list[Any]) -> None:
    """Show localized empty states in each Advanced section."""
    for tab, key in zip(tabs, ("empty_analysis", "empty_detection", "empty_processing", "empty_comparison", "empty_export", "empty_ocr")):
        with tab:
            st.info(t(key))


def _render_advanced(uploaded_files: list[Any]) -> None:
    """Render existing analysis, detection, processing, comparison, export and OCR flows."""
    config = _make_config()
    advanced = st.tabs(tuple(t(key) for key in ("analysis", "detection", "processing", "comparison", "export", "ocr")))
    records = _decode_records(uploaded_files) if uploaded_files else []
    if not records:
        _render_empty_advanced(advanced)
        return

    uploaded, payload, image = records[0]
    image_key = hashlib.sha256(payload).hexdigest()
    source_stem = _safe_download_stem(uploaded.name)
    config_key = json.dumps(asdict(config), sort_keys=True)
    try:
        result = _cached_pipeline(image, image_key, config_key)
    except (TypeError, ValueError, cv2.error) as exc:
        _show_error(t("pipeline_error"), exc, "advanced_pipeline_error")
        return
    info = _cached_image_analysis(image, image_key)

    with advanced[0]:
        st.subheader(t("image_summary"))
        display_image(st, image, t("original"))
        st.dataframe([info], width="stretch")
        figure, _ = plot_histogram(to_grayscale(image))
        st.pyplot(figure, clear_figure=True)
        figure.clear()

    with advanced[1]:
        detection = result.detection
        status_label = {"ok": t("status_ok"), "low_confidence": t("status_low"), "not_found": t("status_missing")}.get(detection.status, t("status_missing"))
        st.write({t("status"): status_label, t("score"): round(detection.score, 4)})
        with st.expander(t("details"), key="advanced_detection_details"):
            st.write({t("detail_reason"): detection.reason})
        display_image(st, _detection_overlay(image, detection), t("detection_overlay"))
        columns = st.columns(2)
        for column, debug_key, label in zip(columns, ("edges", "mask"), ("edge_map", "mask")):
            with column:
                value = detection.debug.get(debug_key)
                if isinstance(value, np.ndarray):
                    display_image(st, value, t(label))
        candidates = detection.debug.get("candidates", [])
        st.caption(f"{t('candidate_count')}: {len(candidates) if hasattr(candidates, '__len__') else 0}")
        if detection.status != "ok":
            st.warning(t("low_help"))

    with advanced[2]:
        st.caption(t("advanced_auto_note"))
        _render_result(result, config, source_stem, "processing")

    with advanced[3]:
        if st.button(t("run_comparisons"), key="run_comparisons"):
            try:
                render_comparisons(st, to_grayscale(image))
            except (TypeError, ValueError, cv2.error) as exc:
                _show_error(t("comparison_error"), exc, "comparison_error_details")
        else:
            st.info(t("comparison_prompt"))

    with advanced[4]:
        _render_result(result, config, source_stem, "export")

    with advanced[5]:
        if not is_available():
            st.info(t("tesseract_missing"))
        else:
            source = st.selectbox(
                t("ocr_source"), _literal_options("ocr_source"), key="ocr_source",
                format_func=lambda value: _option_label("ocr_source", value),
            )
            languages = check_installation().get("missing", [])
            if languages:
                st.warning(t("missing_languages"))
            language = st.selectbox(
                t("ocr_lang"), ("tur+eng", "tur", "eng"), key="ocr_lang",
                format_func=lambda value: t(f"language_{value.replace('+', '_')}") ,
            )
            orient = st.checkbox(t("orientation"), value=True, key="ocr_orient")
            if st.button(t("run_ocr"), key="run_ocr"):
                try:
                    ocr_image, actual_source = select_ocr_input(result, replace(config, ocr_source=source))
                    if ocr_image is None:
                        st.warning(t("no_binary"))
                    else:
                        if actual_source != source:
                            st.info(t("fallback_ocr"))
                        orientation_warning = None
                        if orient:
                            ocr_image, orientation_warning = correct_orientation(ocr_image, lang=language)
                        ocr_result = run_ocr(ocr_image, lang=language)
                        st.session_state["ocr_result"] = ocr_result
                        st.session_state["ocr_result_key"] = (image_key, config_key, source, language, orient)
                        st.session_state["ocr_warning"] = orientation_warning
                except (RuntimeError, TypeError, ValueError, cv2.error) as exc:
                    _show_error(t("ocr_error"), exc, "ocr_error_details")
            ocr_result = st.session_state.get("ocr_result")
            expected_ocr_key = (image_key, config_key, source, language, orient)
            if ocr_result is not None and st.session_state.get("ocr_result_key") == expected_ocr_key:
                if st.session_state.get("ocr_warning"):
                    st.warning(st.session_state["ocr_warning"])
                st.metric(t("ocr_confidence"), f"{ocr_result.mean_conf:.1f}")
                st.text_area(t("ocr_text"), value=ocr_result.text, height=220, key="ocr_text")
                st.download_button(
                    t("advanced_txt_download"), data=ocr_result.text.encode("utf-8"),
                    file_name=f"{source_stem}_ocr.txt", mime="text/plain; charset=utf-8",
                    key="ocr_text_download",
                )


def _render_scan(uploaded_files: list[Any]) -> None:
    """Render the simple automatic scan workflow."""
    if not uploaded_files:
        st.markdown(f'<div class="sd-card"><h3>{t("no_upload")}</h3><p>{t("no_upload_help")}</p></div>', unsafe_allow_html=True)
        columns = st.columns(3)
        for column, title, description in zip(
            columns,
            ("step_upload", "step_review", "step_download"),
            ("step_upload_help", "step_review_help", "step_download_help"),
        ):
            with column, st.container(border=True):
                st.subheader(t(title))
                st.caption(t(description))
        return

    scan_files = sorted(uploaded_files, key=lambda upload: _natural_sort_key(upload.name))
    if len(scan_files) > MAX_UPLOAD_PAGES:
        st.warning(t("page_limit"))
        scan_files = scan_files[:MAX_UPLOAD_PAGES]
    records = _decode_records(scan_files)
    if not records:
        return
    excluded = set(st.session_state.get("scan_excluded_pages", []))
    keyed_records: dict[str, tuple[Any, bytes, np.ndarray, str]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for record in records:
        item, payload, image = record
        image_key = hashlib.sha256(payload).hexdigest()
        pair = (item.name.casefold(), image_key)
        occurrence = occurrences.get(pair, 0)
        occurrences[pair] = occurrence + 1
        page_id = hashlib.sha256(f"{item.name}\0{image_key}\0{occurrence}".encode("utf-8")).hexdigest()
        keyed_records[page_id] = (item, payload, image, image_key)
    keyed_records = {key: value for key, value in keyed_records.items() if key not in excluded}
    if not keyed_records:
        st.info(t("no_upload_help"))
        return
    page_order = [key for key in st.session_state.get("scan_page_order", []) if key in keyed_records]
    page_order.extend(key for key in keyed_records if key not in page_order)
    st.session_state["scan_page_order"] = page_order
    batch_key = hashlib.sha256("\0".join(page_order).encode("ascii")).hexdigest()
    config_json = json.dumps(asdict(TARA_CONFIG), sort_keys=True)
    processed: list[tuple[Any, bytes, np.ndarray, PipelineResult, str, str]] = []
    status = st.status(t("scan_running"), expanded=False)
    try:
        for page_id in page_order:
            item, payload, image, image_key = keyed_records[page_id]
            result = _cached_pipeline(image, image_key, config_json, retain_intermediates=False)
            processed.append((item, payload, image, result, image_key, page_id))
        priorities = {"ok": 0, "low_confidence": 1, "not_found": 2}
        overall_status = max(
            (page[3].detection.status for page in processed),
            key=lambda value: priorities.get(value, 2),
        )
        overall_label = {
            "ok": t("found"),
            "low_confidence": t("low_confidence"),
            "not_found": t("not_found"),
        }.get(overall_status, t("not_found"))
        status.update(label=overall_label, state="complete", expanded=False)
    except (TypeError, ValueError, cv2.error) as exc:
        status.update(label=t("process_error"), state="error", expanded=True)
        _show_error(t("process_error"), exc, "scan_pipeline_error")
        return

    st.subheader(t("file_count").format(count=len(processed)))
    consented = set(st.session_state.get("scan_consented_pages", []))
    for index, page in enumerate(processed):
        item, _, page_image, page_result, _, page_id = page
        page_status = page_result.detection.status
        label = {
            "ok": t("found"),
            "low_confidence": t("low_confidence"),
            "not_found": t("not_found"),
        }.get(page_status, t("not_found"))
        thumb, name, select_col, up_col, down_col, remove_col = st.columns([1, 4, 1.2, 0.8, 0.8, 0.8])
        with thumb:
            display_image(st, page_image, item.name, max_dimension=90)
        with name:
            st.caption(item.name)
            badge_class = {"ok": "sd-ok", "low_confidence": "sd-warn", "not_found": "sd-error"}.get(page_status, "sd-error")
            st.markdown(f'<span class="sd-status {badge_class}">{label}</span>', unsafe_allow_html=True)
        with select_col:
            if st.button(t("select_page"), key=f"scan_select_{page_id[:12]}"):
                st.session_state["scan_selected_index"] = index
                st.rerun()
        with up_col:
            if st.button(t("move_up"), key=f"scan_up_{page_id[:12]}", disabled=index == 0):
                page_order[index - 1], page_order[index] = page_order[index], page_order[index - 1]
                st.session_state["scan_page_order"] = page_order
                st.session_state["scan_selected_index"] = index - 1
                st.rerun()
        with down_col:
            if st.button(t("move_down"), key=f"scan_down_{page_id[:12]}", disabled=index == len(processed) - 1):
                page_order[index + 1], page_order[index] = page_order[index], page_order[index + 1]
                st.session_state["scan_page_order"] = page_order
                st.session_state["scan_selected_index"] = index + 1
                st.rerun()
        with remove_col:
            if st.button(t("remove_page"), key=f"scan_remove_{page_id[:12]}"):
                excluded.add(page_id)
                st.session_state["scan_excluded_pages"] = sorted(excluded)
                st.session_state["scan_page_order"] = [key for key in page_order if key != page_id]
                st.session_state["scan_selected_index"] = max(0, min(index, len(processed) - 2))
                st.rerun()
        if page_status != "ok" and page_id not in consented:
            if st.button(t("approve_page"), key=f"scan_approve_{page_id[:12]}"):
                consented.add(page_id)
                st.session_state["scan_consented_pages"] = sorted(consented)
                st.rerun()

    priorities = {"ok": 0, "low_confidence": 1, "not_found": 2}
    worst = max((page[3].detection.status for page in processed), key=lambda value: priorities.get(value, 2))
    worst_label = {"ok": t("found"), "low_confidence": t("low_confidence"), "not_found": t("not_found")}[worst]
    st.caption(f"{t('worst_status')}: {worst_label}")

    selected = st.session_state.get("scan_selected_index", 0)
    selected = max(0, min(int(selected), len(processed) - 1))
    st.session_state["scan_selected_index"] = selected
    item, payload, image, result, image_key, selected_page_id = processed[selected]
    detection = result.detection
    if overall_status == "ok":
        st.markdown(f'<span class="sd-status sd-ok">{t("found")}</span>', unsafe_allow_html=True)
        st.caption(t("found_help"))
    elif overall_status == "low_confidence":
        st.markdown(f'<span class="sd-status sd-warn">{t("low_confidence")}</span>', unsafe_allow_html=True)
        st.warning(t("low_help"))
    else:
        st.markdown(f'<span class="sd-status sd-error">{t("not_found")}</span>', unsafe_allow_html=True)
        st.warning(t("not_found_help"))
    selected_failed = detection.status != "ok"
    if selected_failed and selected_page_id not in consented:
        st.warning(t("warning_no_rectification"))

    mode_labels = [t(SCAN_LABELS[mode]) for mode in SCAN_MODES]
    mode_label = st.segmented_control(t("scan_mode"), mode_labels, default=mode_labels[0], key="scan_mode")
    selected_mode = SCAN_MODES[mode_labels.index(mode_label)] if mode_label in mode_labels else "color"
    st.caption(t("page_order"))

    source = result.rectified_image if result.rectified_image is not None else image
    if selected_failed and selected_page_id not in consented:
        left, right = st.columns(2)
        with left:
            display_image(st, _detection_overlay(image, detection), t("original"))
        with right:
            st.info(t("not_found_help"))
        return

    try:
        scan_image = _cached_scan(source, image_key, selected_mode)
        pending_pages = [
            page for page in processed
            if page[3].detection.status != "ok" and page[5] not in consented
        ]
        pdf_sources = []
        if not pending_pages:
            for _, _, page_image, page_result, page_key, _ in processed:
                page_source = page_result.rectified_image if page_result.rectified_image is not None else page_image
                pdf_sources.append(_cached_scan(page_source, page_key, selected_mode))
        orientation_label = st.selectbox(
            t("a4_orientation"),
            (t("orientation_auto"), t("orientation_portrait"), t("orientation_landscape")),
            key="scan_a4_orientation",
        )
        orientation = {
            t("orientation_auto"): "auto",
            t("orientation_portrait"): "portrait",
            t("orientation_landscape"): "landscape",
        }[orientation_label]
        pdf_bytes = (
            _cached_pdf(pdf_sources, batch_key, selected_mode, "a4", orientation)
            if pdf_sources else None
        )
    except (TypeError, ValueError, OSError, cv2.error) as exc:
        _show_error(t("scan_error"), exc, "scan_error_details")
        return

    st.session_state["scan_result_key"] = (batch_key, selected_mode)
    if pending_pages:
        st.warning(t("pending_pages"))

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader(t("original"))
        display_image(st, _detection_overlay(image, detection), item.name)
    with right, st.container(border=True):
        st.subheader(t("scan_result"))
        display_image(st, scan_image, t(SCAN_LABELS[selected_mode]))

    buttons = st.columns([2, 1])
    if pdf_bytes is not None:
        with buttons[0]:
            st.download_button(
                t("pdf_download"), data=pdf_bytes,
                file_name="tarama.pdf" if len(pdf_sources) > 1 else f"{_safe_download_stem(item.name)}_tarama.pdf",
                mime="application/pdf", type="primary", width="stretch", key="scan_pdf_download",
            )
    with buttons[1]:
        st.download_button(
            t("png_download"), data=_encode_png(scan_image),
            file_name=f"{_safe_download_stem(item.name)}_tarama.png",
            mime="image/png", width="stretch", key="scan_png_download",
        )


def main() -> None:
    """Render the default scan screen and opt-in Advanced workspace."""
    st.set_page_config(
        page_title="SmartDoc", page_icon="📄", layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    st.markdown(f'<div class="sd-title">{t("brand")}</div><div class="sd-subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)
    view = st.segmented_control(
        t("workspace"), (t("scan"), t("advanced")),
        default=t("scan"), key="main_view", label_visibility="collapsed",
    )
    uploaded_files = st.file_uploader(
        t("upload"), type=("png", "jpg", "jpeg"), accept_multiple_files=True,
        key="document_upload", help=t("upload_help"),
    ) or []
    if view == t("advanced"):
        _render_advanced(uploaded_files)
    else:
        _render_scan(uploaded_files)


if __name__ == "__main__":
    main()
