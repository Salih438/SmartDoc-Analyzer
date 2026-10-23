"""SmartDoc Analyzer Streamlit UI; document algorithms remain in ``src``."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import get_args, get_type_hints

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.comparison_view import render_comparisons
from app.components.image_viewer import display_image
from src.analysis.histogram import plot_histogram
from src.analysis.image_info import get_image_info, load_image_from_bytes
from src.document.detection import DetectionResult, detect_document
from src.pipeline import PipelineConfig, PipelineResult, process_image
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.filtering import median_blur
from src.evaluation.metrics import sharpness_laplacian_var
from src.ocr.engine import check_installation, correct_orientation, is_available, run_ocr


@st.cache_data(show_spinner=False)
def _decode_uploaded_image(data: bytes) -> np.ndarray:
    """Pure decode cache; callers receive a copied array before processing."""
    return load_image_from_bytes(data)


@st.cache_data(show_spinner=False)
def _cached_image_analysis(image: np.ndarray) -> dict[str, object]:
    """Pure image summary cache for repeated Streamlit reruns."""
    information: dict[str, object] = get_image_info(image)
    information["sharpness"] = sharpness_laplacian_var(to_grayscale(image))
    return information


@st.cache_data(show_spinner=False)
def _cached_detection(
    image: np.ndarray, strategy: str, canny_mode: str, denoise: bool
) -> DetectionResult:
    """Cache the deterministic detector result for a specific image/config."""
    detection_input = median_blur(to_grayscale(image), ksize=5) if denoise else image
    return detect_document(detection_input, strategy=strategy, canny_mode=canny_mode)


def _literal_options(field: str) -> tuple[str, ...]:
    """Read selector choices from PipelineConfig's declared Literal type."""
    choices = get_args(get_type_hints(PipelineConfig)[field])
    if not choices or not all(isinstance(choice, str) for choice in choices):
        raise TypeError(f"PipelineConfig.{field} alanında Literal seçenekleri tanımlı değil.")
    return tuple(choices)


def _safe_download_stem(filename: str) -> str:
    """Keep Unicode basename characters while removing unsafe path/name chars."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[\\/<>:\"|?*\x00-\x1f]", "_", stem).strip(" .")
    return cleaned or "smartdoc"


def _make_config() -> PipelineConfig:
    """Build the processing config from current sidebar values."""
    with st.sidebar:
        st.header("İşlem ayarları")
        detect = st.checkbox("Belge sınırını tespit et", value=True, key="cfg_detect")
        strategy = st.selectbox(
            "Tespit stratejisi", _literal_options("detection_strategy"), key="cfg_strategy"
        )
        canny_mode = st.selectbox(
            "Canny modu", _literal_options("canny_mode"), key="cfg_canny"
        )
        detect_denoise = st.checkbox(
            "Tespitten önce Median(5)", value=False, key="cfg_detect_denoise"
        )
        detection_fallback = st.checkbox(
            "Canny başarısızsa threshold yedeğini dene",
            value=False,
            key="cfg_detection_fallback",
        )
        enhance = st.selectbox("İyileştirme", _literal_options("enhance"), key="cfg_enhance")
        denoise = st.selectbox("Gürültü azaltma", _literal_options("denoise"), key="cfg_denoise")
        sharpen = st.selectbox("Keskinleştirme", _literal_options("sharpen"), key="cfg_sharpen")
        binarize = st.selectbox("İkili eşikleme", _literal_options("binarize"), key="cfg_binarize")
        morphology = st.selectbox(
            "Morfoloji", _literal_options("morphology"), key="cfg_morphology"
        )
        allow_unrectified = st.checkbox(
            "Tespit başarısızsa düzeltilmemiş görüntüyle devam etmeyi onaylıyorum",
            value=False,
            key="cfg_allow_unrectified",
            help="Kapalıyken tespit başarısızsa pipeline sonraki görüntü işleme adımlarını çalıştırmaz.",
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
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    if detection.corners is not None:
        cv2.polylines(
            overlay,
            [np.rint(detection.corners).astype(np.int32)],
            True,
            (0, 255, 0),
            3,
        )
    return overlay


def _render_detection(st, image: np.ndarray, detection: DetectionResult) -> None:
    st.subheader("Belge sınırı tespiti")
    st.write({"Durum": detection.status, "Skor": round(detection.score, 4), "Strateji": detection.strategy, "Neden": detection.reason})
    display_image(st, _detection_overlay(image.copy(), detection), "Tespit overlay’i")
    cols = st.columns(2)
    with cols[0]:
        edges = detection.debug.get("edges")
        if isinstance(edges, np.ndarray):
            display_image(st, edges, "Kenar haritası")
    with cols[1]:
        mask = detection.debug.get("mask")
        if isinstance(mask, np.ndarray):
            display_image(st, mask, "Tespit maskesi")
    candidates = detection.debug.get("candidates", [])
    st.caption(f"Aday sayısı: {len(candidates) if hasattr(candidates, '__len__') else 0}")
    if detection.status != "ok":
        st.warning(
            f"Belge güvenilir biçimde tespit edilemedi ({detection.status}). "
            "Düzeltilmemiş tam görüntüyle devam etmek için sol menüde açık onay gerekir."
        )


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG çıktısı bellekte kodlanamadı.")
    return buffer.tobytes()


def _render_result(st, result: PipelineResult, config: PipelineConfig, source_stem: str) -> None:
    st.write({"Durum": result.detection.status, "Düzeltilmiş": result.rectified})
    for warning in result.warnings:
        st.warning(warning)
    if not result.rectified and result.detection.status != "ok":
        st.info("Güvenilir belge warp’ı uygulanmadı; onay kapalıysa işleme çıktısı üretilmez.")
        return
    original_column, rectified_column = st.columns(2)
    with original_column:
        display_image(st, result.original, "Orijinal", max_dimension=1400)
    with rectified_column:
        if result.rectified and result.gray is not None:
            display_image(st, result.gray, "Perspektif düzeltilmiş gri")
        elif result.gray is not None:
            display_image(st, result.gray, "Düzeltilmemiş gri (kullanıcı onayıyla)")
    for name, image in (("İyileştirilmiş", result.enhanced), ("Gürültüsü azaltılmış", result.denoised), ("Son ikili çıktı", result.binary)):
        if image is not None:
            display_image(st, image, name)
    st.subheader("Adımlar")
    st.dataframe(result.steps, use_container_width=True)
    st.subheader("Analiz: önce / sonra")
    st.dataframe(
        [{"Aşama": "Önce", **result.info_before}, {"Aşama": "Sonra", **result.info_after}],
        use_container_width=True,
    )
    if result.binary is not None:
        st.download_button(
            "Son çıktıyı PNG indir",
            data=_encode_png(result.binary),
            file_name=f"{source_stem}_sonuc.png",
            mime="image/png",
        )
    metadata = {"config": asdict(config), "steps": result.steps, "warnings": result.warnings}
    st.download_button(
        "Config + adımları JSON indir",
        data=json.dumps(metadata, ensure_ascii=False, indent=2),
        file_name=f"{source_stem}_pipeline.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="SmartDoc Analyzer", page_icon="📄", layout="wide")
    st.title("SmartDoc Analyzer")
    st.caption("Klasik OpenCV + NumPy belge analizi ve ön işleme")
    uploaded = st.sidebar.file_uploader(
        "Tek belge görüntüsü yükle", type=("png", "jpg", "jpeg"), key="document_upload"
    )
    config = _make_config()

    tabs = st.tabs(("Analiz", "Tespit", "İşleme", "Karşılaştırma", "Dışa aktar", "OCR"))
    if uploaded is None:
        with tabs[0]:
            st.info("Başlamak için PNG veya JPG/JPEG belge görüntüsü yükleyin.")
        with tabs[1]:
            st.info("Tespit debug bilgisi yüklemeden sonra görünür.")
        with tabs[2]:
            st.info("Pipeline yüklenen tam çözünürlüklü görüntü üzerinde çalışır.")
        with tabs[3]:
            st.info("Karşılaştırmalar için önce görüntü yükleyin.")
        with tabs[4]:
            st.info("İşleme sonrası PNG ve JSON indirilebilir.")
        with tabs[5]:
            st.info("OCR için bir görüntü yükleyin.")
        return

    payload = uploaded.getvalue()
    image_key = hashlib.sha256(payload).hexdigest()
    source_stem = _safe_download_stem(uploaded.name)
    try:
        image = _decode_uploaded_image(payload).copy()
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return
    if image.dtype != np.uint8:
        st.error("Yalnızca uint8 görüntüler desteklenir.")
        return

    info = _cached_image_analysis(image.copy())
    try:
        detection = _cached_detection(
            image.copy(), config.detection_strategy, config.canny_mode, config.detect_denoise
        )
    except (TypeError, ValueError, cv2.error) as exc:
        st.error(f"Belge tespiti çalıştırılamadı: {exc}")
        return

    with tabs[0]:
        st.subheader("Orijinal görüntü")
        display_image(st, image.copy(), uploaded.name or "Yüklenen görüntü")
        st.dataframe([info], use_container_width=True)
        gray = to_grayscale(image)
        figure, _ = plot_histogram(gray)
        st.pyplot(figure, clear_figure=True)
        figure.clear()

    with tabs[1]:
        _render_detection(st, image, detection)

    config_key = json.dumps(asdict(config), sort_keys=True)
    with tabs[2]:
        if st.button("Pipeline çalıştır", key="run_pipeline"):
            try:
                st.session_state["pipeline_result"] = process_image(image.copy(), config)
                st.session_state["pipeline_result_key"] = (image_key, config_key)
            except (TypeError, ValueError, cv2.error) as exc:
                st.error(f"Pipeline çalıştırılamadı: {exc}")
        stored_result = st.session_state.get("pipeline_result")
        stored_key = st.session_state.get("pipeline_result_key")
        if stored_result is None or stored_key != (image_key, config_key):
            st.info("Ayarlar uygulansın diye ‘Pipeline çalıştır’ düğmesine basın.")
        else:
            _render_result(st, stored_result, config, source_stem)

    with tabs[3]:
        if st.button("Karşılaştırmaları hesapla", key="run_comparisons"):
            try:
                render_comparisons(st, to_grayscale(image))
            except (TypeError, ValueError, cv2.error) as exc:
                st.warning(f"Karşılaştırma için bu görüntü uygun değil: {exc}")
        else:
            st.info("Global/Otsu/Adaptive ve filtre karşılaştırmalarını çalıştırmak için düğmeye basın.")

    with tabs[4]:
        current = st.session_state.get("pipeline_result")
        current_key = st.session_state.get("pipeline_result_key")
        if current is not None and current_key == (image_key, config_key):
            _render_result(st, current, config, source_stem)
        else:
            st.info("İndirme için önce pipeline çalıştırın.")

    with tabs[5]:
        if not is_available():
            installation = check_installation()
            st.info(str(installation.get("error", "Tesseract bulunamadı. OCR için Tesseract'ı kurup PATH'e ekleyin.")))
        else:
            language_status = check_installation()
            missing_languages = language_status.get("missing", [])
            if missing_languages:
                st.warning(f"Eksik Tesseract dil verileri: {', '.join(missing_languages)}")
            orient = st.checkbox("OCR öncesi yönü düzelt", value=True, key="ocr_orient")
            if st.button("OCR çalıştır", key="run_ocr"):
                try:
                    current = st.session_state.get("pipeline_result")
                    if current is None or st.session_state.get("pipeline_result_key") != (image_key, config_key):
                        current = process_image(image.copy(), config)
                    if current.binary is None:
                        st.warning("İşlenecek ikili görüntü yok; tespit/işleme ayarlarını ve devam onayını kontrol edin.")
                    else:
                        ocr_image = current.binary
                        orientation_warning = None
                        if orient:
                            ocr_image, orientation_warning = correct_orientation(ocr_image)
                        ocr_result = run_ocr(ocr_image)
                        st.session_state["ocr_result"] = ocr_result
                        st.session_state["ocr_result_key"] = (image_key, config_key)
                        st.session_state["ocr_warning"] = orientation_warning
                except (RuntimeError, TypeError, ValueError, cv2.error) as exc:
                    st.error(str(exc))
            result_ocr = st.session_state.get("ocr_result")
            if result_ocr is not None and st.session_state.get("ocr_result_key") == (image_key, config_key):
                if st.session_state.get("ocr_warning"):
                    st.warning(st.session_state["ocr_warning"])
                st.metric("Ortalama OCR güveni", f"{result_ocr.mean_conf:.1f}")
                st.text_area("Tanınan metin", value=result_ocr.text, height=220, key="ocr_text")
                st.download_button(
                    "Metni TXT indir", data=result_ocr.text.encode("utf-8"),
                    file_name=f"{source_stem}_ocr.txt", mime="text/plain; charset=utf-8",
                )


if __name__ == "__main__":
    main()
