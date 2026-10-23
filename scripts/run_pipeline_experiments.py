"""Reproducible pipeline experiments; metric tables are derived from their CSVs."""

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

from scripts.run_experiments import create_synthetic_document
from src.document.perspective import order_corners
from src.evaluation.metrics import f1_score, precision, recall, rectification_ssim, quad_iou, corner_error
from src.evaluation.noise import add_gaussian_noise, simulate_uneven_illumination
from src.evaluation.scenes import make_scene
from src.pipeline import PipelineConfig, process_image


CONFIGURATIONS = {
    "yalniz_otsu": PipelineConfig(binarize="otsu"),
    "yalniz_adaptive": PipelineConfig(binarize="adaptive_mean"),
    "denoise_adaptive": PipelineConfig(denoise="median", binarize="adaptive_mean"),
    "auto": PipelineConfig(enhance="auto", denoise="auto", binarize="adaptive_gaussian"),
}
CONDITIONS = (
    ("frontal_clean", 0.0, 0.0),
    ("frontal_easy", 0.7, 10.0),
    ("frontal_easy", 0.9, 10.0),
    ("low_contrast_bg", 0.7, 10.0),
    ("low_contrast_hard", 0.9, 10.0),
    ("rotated_30", 0.7, 10.0),
    ("strong_perspective", 0.9, 10.0),
    ("noisy_shadow", 0.9, 20.0),
)
SEEDS = (0, 1, 2)


def _aligned_gt_mask(
    scene_mask: np.ndarray, detected_corners: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Warp text GT with the same detected homography used by the pipeline."""
    width, height = size
    source = order_corners(detected_corners).astype(np.float32)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        scene_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _scene_text_ground_truth(
    text_mask: np.ndarray, corners: np.ndarray, scene_shape: tuple[int, int]
) -> np.ndarray:
    """Place the binary document GT by geometry only, unaffected by scene light/noise."""
    source_height, source_width = text_mask.shape[:2]
    source = np.array(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, order_corners(corners).astype(np.float32))
    height, width = scene_shape
    warped = cv2.warpPerspective(
        text_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.where(warped >= 128, 255, 0).astype(np.uint8)


def _best_text_metrics(pred: np.ndarray | None, gt: np.ndarray) -> tuple[float, float, float]:
    if pred is None:
        return 0.0, 0.0, 0.0
    scores: list[tuple[float, float, float]] = []
    for turns in range(4):
        oriented = np.rot90(gt, turns)
        aligned = cv2.resize(oriented, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        scores.append((precision(aligned, pred), recall(aligned, pred), f1_score(aligned, pred)))
    return max(scores, key=lambda item: item[2])


def _apply_condition(
    scene_bgr: np.ndarray, seed: int, shadow_strength: float, noise_sigma: float
) -> np.ndarray:
    gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    if shadow_strength:
        gray = simulate_uneven_illumination(gray, strength=shadow_strength, seed=seed + 200)
    if noise_sigma:
        gray = add_gaussian_noise(gray, sigma=noise_sigma, seed=seed + 400)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"CSV için satır üretilmedi: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_detection_denoise_comparison(doc: np.ndarray, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        scene, gt = make_scene(doc, "low_contrast_bg", seed)
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        gray = simulate_uneven_illumination(gray, strength=0.7, seed=seed + 200)
        gray = add_gaussian_noise(gray, sigma=10.0, seed=seed + 400)
        input_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for label, denoise in (("raw", False), ("median_5", True)):
            started = time.perf_counter()
            pipeline_result = process_image(
                input_bgr,
                PipelineConfig(
                    detection_strategy="canny", canny_mode="fixed",
                    detect_denoise=denoise, binarize="none",
                ),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            detection = pipeline_result.detection
            error_px, error_ratio = corner_error(gt, detection.corners)
            rows.append({
                "Senaryo": "low_contrast_bg_shadow_noise",
                "Seed": seed,
                "Yontem": label,
                "Shadow_strength": 0.7,
                "Noise_sigma": 10.0,
                "Durum": detection.status,
                "KoseHata_px": f"{error_px:.3f}",
                "KoseHata_oran": f"{error_ratio:.6f}",
                "IoU": f"{quad_iou(gt, detection.corners, gray.shape):.6f}",
                "Sure_ms": f"{elapsed_ms:.3f}",
                "Neden": detection.reason,
            })
    _write_rows(output_dir / "pipeline_detection_denoise.csv", rows)


def run() -> None:
    doc, text_mask = create_synthetic_document()
    output_dir = ROOT / "results"
    scene_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []

    for scenario, shadow_strength, noise_sigma in CONDITIONS:
        for seed in SEEDS:
            scene, corners = make_scene(doc, scenario, seed)
            scene_gt = _scene_text_ground_truth(text_mask, corners, scene.shape[:2])
            input_bgr = _apply_condition(scene, seed, shadow_strength, noise_sigma)

            # Rectification is a detector/scene property, not a binarizer metric.
            scene_result = process_image(input_bgr, CONFIGURATIONS["yalniz_otsu"])
            scene_ssim = (
                rectification_ssim(doc, scene_result.gray)
                if scene_result.rectified and scene_result.gray is not None
                else float("nan")
            )
            scene_rows.append({
                "Senaryo": scenario,
                "Seed": seed,
                "Shadow_strength": shadow_strength,
                "Noise_sigma": noise_sigma,
                "Durum": scene_result.detection.status,
                "Duzeltildi": scene_result.rectified,
                "SSIM_duzeltilmis": f"{scene_ssim:.6f}" if np.isfinite(scene_ssim) else "",
                "Neden": scene_result.detection.reason,
            })

            for name, config in CONFIGURATIONS.items():
                result = process_image(input_bgr, config)
                if result.binary is not None and result.detection.corners is not None:
                    aligned_gt = _aligned_gt_mask(
                        scene_gt, result.detection.corners,
                        (result.binary.shape[1], result.binary.shape[0]),
                    )
                    p_value, r_value, text_f1 = _best_text_metrics(result.binary, aligned_gt)
                else:
                    p_value = r_value = text_f1 = 0.0
                steps = {str(step["ad"]): step["parametreler"] for step in result.steps}
                enhance_params = steps.get("enhance", {})
                denoise_params = steps.get("denoise", {})
                before = result.info_before
                config_rows.append({
                    "Senaryo": scenario,
                    "Seed": seed,
                    "Shadow_strength": shadow_strength,
                    "Noise_sigma": noise_sigma,
                    "Konfigurasyon": name,
                    "Durum": result.detection.status,
                    "Duzeltildi": result.rectified,
                    "Precision": f"{p_value:.6f}",
                    "Recall": f"{r_value:.6f}",
                    "Metin_F1": f"{text_f1:.6f}",
                    "Auto_enhance": enhance_params.get("method", "") if name == "auto" else "",
                    "Auto_denoise": denoise_params.get("method", "") if name == "auto" else "",
                    "Noise_sigma_tahmin": denoise_params.get("noise_sigma_estimate", "") if name == "auto" else "",
                    "Parlaklik": f"{float(before.get('mean_brightness', float('nan'))):.4f}",
                    "Kontrast": f"{float(before.get('std_contrast', float('nan'))):.4f}",
                    "Uyarilar": " | ".join(result.warnings),
                    "Adim_sureleri_ms": ";".join(
                        f"{step['ad']}={float(step['sure_ms']):.2f}" for step in result.steps
                    ),
                })

    scene_csv = output_dir / "pipeline_rectification_metrics.csv"
    config_csv = output_dir / "pipeline_experiments.csv"
    _write_rows(scene_csv, scene_rows)
    _write_rows(config_csv, config_rows)
    _write_detection_denoise_comparison(doc, output_dir)

    # Grouped display rows are computed from the saved config CSV, not source buffers.
    saved = _read_rows(config_csv)
    summary: list[dict[str, object]] = []
    groups = sorted({(row["Senaryo"], row["Konfigurasyon"]) for row in saved})
    for scenario, config_name in groups:
        group = [row for row in saved if row["Senaryo"] == scenario and row["Konfigurasyon"] == config_name]
        summary.append({
            "Senaryo": scenario,
            "Konfigurasyon": config_name,
            "N": len(group),
            "ok": sum(row["Durum"] == "ok" for row in group),
            "Precision_ortalama": f"{np.mean([float(row['Precision']) for row in group]):.5f}",
            "Recall_ortalama": f"{np.mean([float(row['Recall']) for row in group]):.5f}",
            "Metin_F1_ortalama": f"{np.mean([float(row['Metin_F1']) for row in group]):.5f}",
        })
    _write_rows(output_dir / "pipeline_experiments_summary.csv", summary)

    print("Pipeline ölçüm özeti (pipeline_experiments.csv üzerinden):")
    for row in summary:
        print(" | ".join(str(row[key]) for key in row))
    print(f"Sahne başına SSIM kayıtları: {len(_read_rows(scene_csv))}; CSV: {scene_csv}")
    print(f"Metin metrik satırları: {len(saved)}; CSV: {config_csv}")
    print(f"Temiz frontal F1 (CSV): {next((row['Metin_F1'] for row in saved if row['Senaryo'] == 'frontal_clean' and row['Konfigurasyon'] == 'yalniz_otsu' and row['Seed'] == '0'), 'yok')}")


if __name__ == "__main__":
    run()
