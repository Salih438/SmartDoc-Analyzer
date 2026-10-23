"""Seed'li sentetik OCR koşuları; her ölçüm satırı CSV'ye kaydedilir."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.noise import add_gaussian_noise, simulate_uneven_illumination
from src.evaluation.ocr_document import make_ocr_document
from src.evaluation.scenes import make_scene
from src.evaluation.text_metrics import cer, wer
from src.ocr.engine import check_installation, is_available, run_ocr
from src.pipeline import PipelineConfig, process_image


CONFIGS = {
    "gri": PipelineConfig(detect=False, binarize="none"),
    "otsu": PipelineConfig(detect=False, binarize="otsu"),
    "adaptive": PipelineConfig(detect=False, binarize="adaptive_gaussian"),
    "denoise_adaptive": PipelineConfig(detect=False, denoise="median", binarize="adaptive_gaussian"),
    "auto": PipelineConfig(detect=False, enhance="auto", denoise="auto", binarize="adaptive_gaussian"),
}


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["Senaryo", "Konfig", "Gölge", "Sigma", "Seed", "CER", "WER", "MeanConf", "Sure_ms", "Durum"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(
    image: np.ndarray, truth: str, scenario: str, config_name: str,
    shadow: float, sigma: float, seed: int, config: PipelineConfig,
) -> dict[str, object]:
    processed = process_image(image.copy(), config)
    detection_status = processed.detection.status if config.detect else "uygulanmadi"
    if config.detect and (processed.detection.status != "ok" or not processed.rectified):
        return {"Senaryo": scenario, "Konfig": config_name, "Gölge": shadow, "Sigma": sigma,
                "Seed": seed, "CER": "", "WER": "", "MeanConf": "", "Sure_ms": "",
                "Tespit_durumu": detection_status, "Durum": f"tespit_{processed.detection.status}"}
    source = processed.binary if processed.binary is not None else processed.gray
    if source is None:
        return {"Senaryo": scenario, "Konfig": config_name, "Gölge": shadow, "Sigma": sigma,
                "Seed": seed, "CER": "", "WER": "", "MeanConf": "", "Sure_ms": "",
                "Tespit_durumu": detection_status, "Durum": "çıktı_yok"}
    if not is_available():
        return {"Senaryo": scenario, "Konfig": config_name, "Gölge": shadow, "Sigma": sigma,
                "Seed": seed, "CER": "", "WER": "", "MeanConf": "", "Sure_ms": "",
                "Tespit_durumu": detection_status, "Durum": "tesseract_kurulu_değil"}
    try:
        result = run_ocr(source)
    except RuntimeError as exc:
        return {"Senaryo": scenario, "Konfig": config_name, "Gölge": shadow, "Sigma": sigma,
                "Seed": seed, "CER": "", "WER": "", "MeanConf": "", "Sure_ms": "",
                "Tespit_durumu": detection_status, "Durum": f"ocr_hatası:{exc}"}
    return {"Senaryo": scenario, "Konfig": config_name, "Gölge": shadow, "Sigma": sigma,
            "Seed": seed, "CER": f"{cer(truth, result.text):.6f}",
            "WER": f"{wer(truth, result.text):.6f}", "MeanConf": f"{result.mean_conf:.3f}",
            "Sure_ms": f"{result.duration_ms:.3f}", "Tespit_durumu": detection_status, "Durum": "ok"}


def run() -> None:
    document, _, truth = make_ocr_document()
    installation = check_installation()
    print(
        f"Tesseract: exe={installation.get('executable')}; "
        f"version={installation.get('version')}; languages={installation.get('languages')}; "
        f"missing={installation.get('missing')}; error={installation.get('error', '')}"
    )
    rows: list[dict[str, object]] = []
    for shadow in (0.0, 0.7):
        for sigma in (0.0, 15.0, 25.0):
            for seed in range(3):
                # Both direct-page and make_scene paths share the exact seed/condition.
                plain = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
                scene, _ = make_scene(document, "frontal_clean", seed)
                for source_name, source in (("belge", plain), ("make_scene", cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))):
                    conditioned = source
                    if shadow:
                        conditioned = simulate_uneven_illumination(conditioned, strength=shadow, seed=seed + 50)
                    if sigma:
                        conditioned = add_gaussian_noise(conditioned, sigma=sigma, seed=seed + 100)
                    input_bgr = cv2.cvtColor(conditioned, cv2.COLOR_GRAY2BGR)
                    for config_name, config in CONFIGS.items():
                        active_config = config
                        if source_name == "make_scene":
                            active_config = PipelineConfig(
                                detect=True, detection_strategy="canny", canny_mode="fixed",
                                enhance=config.enhance, denoise=config.denoise,
                                sharpen=config.sharpen, binarize=config.binarize,
                                morphology=config.morphology,
                            )
                        rows.append(_run_one(
                            input_bgr, truth, source_name, config_name, shadow, sigma, seed,
                            active_config,
                        ))

    for scenario in ("rotated_90", "rotated_30"):
        for seed in range(3):
            scene, _ = make_scene(document, scenario, seed)
            for osd_enabled in (False, True):
                config = PipelineConfig(binarize="otsu")
                pipeline = process_image(scene.copy(), config)
                base = {"Senaryo": scenario, "Konfig": f"otsu_osd_{osd_enabled}", "Gölge": 0,
                        "Sigma": 0, "Seed": seed, "CER": "", "WER": "", "MeanConf": "",
                        "Sure_ms": "", "Tespit_durumu": pipeline.detection.status, "Durum": "tespit_yok"}
                if pipeline.detection.status == "ok" and pipeline.binary is not None and is_available():
                    try:
                        ocr_input = pipeline.binary
                        if osd_enabled:
                            from src.ocr.engine import correct_orientation
                            ocr_input, _ = correct_orientation(ocr_input)
                        started = time.perf_counter()
                        result = run_ocr(ocr_input)
                        base.update({"CER": f"{cer(truth, result.text):.6f}", "WER": f"{wer(truth, result.text):.6f}",
                                     "MeanConf": f"{result.mean_conf:.3f}",
                                     "Sure_ms": f"{(time.perf_counter() - started) * 1000:.3f}", "Durum": "ok"})
                    except RuntimeError as exc:
                        base["Durum"] = f"ocr_hatası:{exc}"
                elif not is_available():
                    base["Durum"] = "tesseract_kurulu_değil"
                rows.append(base)
    output = ROOT / "results" / "ocr_experiments.csv"
    _write(output, rows)
    saved = []
    with output.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    summary: list[dict[str, object]] = []
    keys = ("Senaryo", "Konfig", "Gölge", "Sigma")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in saved:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    for key, group in sorted(groups.items()):
        recognized = [row for row in group if row["Durum"] == "ok" and row["CER"]]
        detection_rows = [row for row in group if row["Tespit_durumu"] != "uygulanmadi"]
        summary.append({
            **dict(zip(keys, key)), "N": len(group),
            "OCR_ok_orani": f"{len(recognized) / len(group):.6f}",
            "Tespit_ok_orani": f"{sum(row['Tespit_durumu'] == 'ok' for row in detection_rows) / len(detection_rows):.6f}" if detection_rows else "",
            "CER_ortalama": f"{np.mean([float(row['CER']) for row in recognized]):.6f}" if recognized else "",
            "WER_ortalama": f"{np.mean([float(row['WER']) for row in recognized]):.6f}" if recognized else "",
            "MeanConf_ortalama": f"{np.mean([float(row['MeanConf']) for row in recognized]):.3f}" if recognized else "",
        })
    _write(ROOT / "results" / "ocr_experiments_summary.csv", summary)
    print(f"OCR deneyi: {len(rows)} satir; CSV={output}")
    if not is_available():
        print("Tesseract kurulu degil; OCR metrikleri bos ve durum tesseract_kurulu_degil.")


if __name__ == "__main__":
    run()
