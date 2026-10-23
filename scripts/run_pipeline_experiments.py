"""Reproducible pipeline experiments; metric tables are derived from their CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiments import create_synthetic_document
from src.document.perspective import compute_output_size, four_point_transform, order_corners
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
CONDITIONS = tuple(
    ("frontal_clean", shadow, noise)
    for shadow in (0.0, 0.4, 0.7, 0.9)
    for noise in (0.0, 15.0, 25.0)
) + (("strong_perspective", 0.0, 0.0),)
SEEDS = (0, 1, 2)


def _aligned_gt_mask(
    scene_mask: np.ndarray, gt_corners: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """GT dörtgen homografisiyle metin maskesini sonuç boyutuna warp eder.

    Problem: resize ve dört rotasyon, perspektifli sahnede GT ile çıktı arasındaki
    geometrik hizayı ölçmez. Algoritma, sahne GT köşelerini sıralayıp hedef raster
    köşelerine getiren homografiyi maskeye uygular. Parametreler çıktı (w,h)'dir.
    Tespit köşeleri yanlışsa sonuç F1'i tespit ve binarizasyon etkilerini birlikte
    içerir; bu ölçüm tespit geometrisini ayrı CSV sütunlarında da raporlar.
    """
    width, height = size
    source = order_corners(gt_corners).astype(np.float32)
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


def _text_metrics(pred: np.ndarray | None, gt: np.ndarray) -> tuple[float, float, float]:
    if pred is None:
        return 0.0, 0.0, 0.0
    if gt.shape != pred.shape:
        raise ValueError(f"Homografik GT ile çıktı boyutu uyuşmuyor: {gt.shape} != {pred.shape}")
    return precision(gt, pred), recall(gt, pred), f1_score(gt, pred)


def _tolerant_f1(pred: np.ndarray, gt: np.ndarray, radius: int) -> tuple[float, float, float]:
    """Pixel precision/recall/F1 with symmetric disk tolerance in pixels."""
    if pred.shape != gt.shape:
        raise ValueError("Toleranslı F1 maskeleri aynı boyutta olmalı")
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_bool, gt_bool = pred > 0, gt > 0
    near_gt = cv2.dilate(gt_bool.astype(np.uint8), kernel) > 0
    near_pred = cv2.dilate(pred_bool.astype(np.uint8), kernel) > 0
    p = float(np.count_nonzero(pred_bool & near_gt)) / max(1, int(np.count_nonzero(pred_bool)))
    r = float(np.count_nonzero(gt_bool & near_pred)) / max(1, int(np.count_nonzero(gt_bool)))
    return p, r, 2 * p * r / max(p + r, 1e-12)


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
    for scenario in ("noisy_shadow", "frontal_easy"):
        for sigma in (10.0, 25.0, 40.0):
            for seed in range(5):
                scene, gt = make_scene(doc, scenario, seed)
                gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
                gray = add_gaussian_noise(gray, sigma=sigma, seed=seed + 400)
                raw_detection, raw_ms = _timed_detect(gray, False)
                median_detection, median_ms = _timed_detect(gray, True)
                for label, detection, elapsed_ms in (
                    ("raw", raw_detection, raw_ms),
                    ("median_5", median_detection, median_ms),
                ):
                    error_px, error_ratio = corner_error(gt, detection.corners)
                    rows.append({
                        "Senaryo": scenario,
                        "Seed": seed,
                        "Yontem": label,
                        "Shadow_strength": "scene_ellipse_max30" if scenario == "noisy_shadow" else 0.0,
                        "Noise_sigma": sigma,
                        "Durum": detection.status,
                        "KoseHata_px": f"{error_px:.3f}",
                        "KoseHata_oran": f"{error_ratio:.6f}",
                        "IoU": f"{quad_iou(gt, detection.corners, gray.shape):.6f}",
                        "Skor": f"{detection.score:.6f}",
                        "Skor_bilesenleri": json.dumps(
                            detection.debug.get("selected_score_components", {}),
                            ensure_ascii=False, sort_keys=True,
                        ),
                        "Sure_ms": f"{elapsed_ms:.3f}",
                        "Neden": detection.reason,
                    })
    _write_rows(output_dir / "pipeline_detection_denoise.csv", rows)


def _timed_detect(gray: np.ndarray, median: bool):
    from src.document.detection import detect_document
    from src.preprocessing.filtering import median_blur

    source = median_blur(gray, ksize=5) if median else gray
    started = time.perf_counter()
    result = detect_document(source, strategy="canny", canny_mode="fixed")
    return result, (time.perf_counter() - started) * 1000.0


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
            clean_scene, _ = make_scene(doc, scenario, seed)
            if scene_result.detection.status == "ok" and scene_result.detection.corners is not None:
                try:
                    clean_warp, _ = four_point_transform(clean_scene, scene_result.detection.corners)
                    scene_ssim = rectification_ssim(doc, clean_warp)
                except ValueError:
                    scene_ssim = float("nan")
            else:
                scene_ssim = float("nan")
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
                        scene_gt, corners,
                        (result.binary.shape[1], result.binary.shape[0]),
                    )
                    p_value, r_value, text_f1 = _text_metrics(result.binary, aligned_gt)
                    tol1_p, tol1_r, tol1_f1 = _tolerant_f1(result.binary, aligned_gt, 1)
                    tol2_p, tol2_r, tol2_f1 = _tolerant_f1(result.binary, aligned_gt, 2)
                else:
                    p_value = r_value = text_f1 = 0.0
                    tol1_p = tol1_r = tol1_f1 = tol2_p = tol2_r = tol2_f1 = 0.0
                steps = {str(step["ad"]): step["parametreler"] for step in result.steps}
                enhance_params = steps.get("enhance", {})
                denoise_params = steps.get("denoise", {})
                before = result.info_before
                gt_w, gt_h = compute_output_size(corners)
                predicted_size = (
                    compute_output_size(result.detection.corners)
                    if result.detection.corners is not None else (0, 0)
                )
                corner_error_px, corner_error_ratio = corner_error(corners, result.detection.corners)
                detection_iou = quad_iou(corners, result.detection.corners, scene.shape[:2])
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
                    "Metin_F1_tolerans_1px": f"{tol1_f1:.6f}",
                    "Precision_tolerans_1px": f"{tol1_p:.6f}",
                    "Recall_tolerans_1px": f"{tol1_r:.6f}",
                    "Metin_F1_tolerans_2px": f"{tol2_f1:.6f}",
                    "Precision_tolerans_2px": f"{tol2_p:.6f}",
                    "Recall_tolerans_2px": f"{tol2_r:.6f}",
                    "Belge_en_boy": f"{doc.shape[1] / doc.shape[0]:.6f}",
                    "GT_cikti_en_boy": f"{gt_w / gt_h:.6f}",
                    "Tespit_cikti_en_boy": f"{predicted_size[0] / predicted_size[1]:.6f}" if predicted_size[1] else "",
                    "GT_cikti_boyut": f"{gt_w}x{gt_h}",
                    "Tespit_cikti_boyut": f"{predicted_size[0]}x{predicted_size[1]}" if predicted_size[1] else "",
                    "KoseHata_px": f"{corner_error_px:.3f}",
                    "KoseHata_oran": f"{corner_error_ratio:.6f}",
                    "Tespit_IoU": f"{detection_iou:.6f}",
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
    auto_rows = [row for row in saved if row["Konfigurasyon"] == "auto" and row["Noise_sigma_tahmin"]]
    noise_summary: list[dict[str, object]] = []
    for sigma in sorted({row["Noise_sigma"] for row in auto_rows}, key=float):
        group = [row for row in auto_rows if row["Noise_sigma"] == sigma]
        noise_summary.append({
            "Gercek_sigma": sigma,
            "Tahmin_sigma_ortalama": f"{np.mean([float(row['Noise_sigma_tahmin']) for row in group]):.4f}",
            "Tahmin_sigma_medyan": f"{np.median([float(row['Noise_sigma_tahmin']) for row in group]):.4f}",
            "Parlaklik_ortalama": f"{np.mean([float(row['Parlaklik']) for row in group]):.3f}",
            "Kontrast_ortalama": f"{np.mean([float(row['Kontrast']) for row in group]):.3f}",
            "N": len(group),
        })
    _write_rows(output_dir / "pipeline_noise_estimates.csv", noise_summary)
    summary: list[dict[str, object]] = []
    groups = sorted({
        (row["Senaryo"], row["Shadow_strength"], row["Noise_sigma"], row["Konfigurasyon"])
        for row in saved
    })
    for scenario, shadow_strength, noise_sigma, config_name in groups:
        group = [
            row for row in saved
            if row["Senaryo"] == scenario
            and row["Shadow_strength"] == shadow_strength
            and row["Noise_sigma"] == noise_sigma
            and row["Konfigurasyon"] == config_name
        ]
        successful = [row for row in group if row["Durum"] == "ok" and row["Duzeltildi"] == "True"]
        auto_rows = [row for row in group if row["Konfigurasyon"] == "auto"]
        summary.append({
            "Senaryo": scenario,
            "Shadow_strength": shadow_strength,
            "Noise_sigma": noise_sigma,
            "Konfigurasyon": config_name,
            "N": len(group),
            "ok": len(successful),
            "Ok_orani": f"{len(successful) / len(group):.5f}",
            "Precision_ortalama": f"{np.mean([float(row['Precision']) for row in successful]):.5f}" if successful else "",
            "Recall_ortalama": f"{np.mean([float(row['Recall']) for row in successful]):.5f}" if successful else "",
            "Metin_F1_ortalama": f"{np.mean([float(row['Metin_F1']) for row in successful]):.5f}" if successful else "",
            "Auto_enhance": ",".join(sorted({row["Auto_enhance"] for row in auto_rows if row["Auto_enhance"]})),
            "Auto_denoise": ",".join(sorted({row["Auto_denoise"] for row in auto_rows if row["Auto_denoise"]})),
            "Noise_sigma_tahmin_ortalama": f"{np.mean([float(row['Noise_sigma_tahmin']) for row in auto_rows if row['Noise_sigma_tahmin']]):.3f}" if any(row["Noise_sigma_tahmin"] for row in auto_rows) else "",
            "Parlaklik_ortalama": f"{np.mean([float(row['Parlaklik']) for row in auto_rows]):.3f}" if auto_rows else "",
            "Kontrast_ortalama": f"{np.mean([float(row['Kontrast']) for row in auto_rows]):.3f}" if auto_rows else "",
        })
    _write_rows(output_dir / "pipeline_experiments_summary.csv", summary)

    print("Pipeline ölçüm özeti (pipeline_experiments.csv üzerinden):")
    for row in summary:
        print(" | ".join(str(row[key]) for key in row))
    print(f"Sahne başına SSIM kayıtları: {len(_read_rows(scene_csv))}; CSV: {scene_csv}")
    print(f"Metin metrik satırları: {len(saved)}; CSV: {config_csv}")
    print(f"Gerçek/tahmin gürültü tablosu: {output_dir / 'pipeline_noise_estimates.csv'}")
    print(f"Temiz frontal F1 (CSV): {next((row['Metin_F1'] for row in saved if row['Senaryo'] == 'frontal_clean' and row['Konfigurasyon'] == 'yalniz_otsu' and row['Seed'] == '0'), 'yok')}")


if __name__ == "__main__":
    run()
