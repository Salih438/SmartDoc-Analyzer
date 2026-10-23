"""Measure page-boundary contrast and the opt-in Canny-to-threshold fallback."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_detection_experiments import _detection_document, _overlay
from src.analysis.image_info import save_image
from src.document.detection import detect_document
from src.evaluation.metrics import corner_error, quad_iou
from src.evaluation.scenes import make_scene
from src.pipeline import PipelineConfig, process_image


CONTRASTS = (10, 20, 30, 40, 60, 80)
SEEDS = range(5)
STRATEGIES = (("canny", "fixed"), ("canny", "auto"), ("canny", "otsu"), ("threshold", "fixed"))
FALLBACK_SCENARIOS = ("low_contrast_bg", "low_contrast_hard", "clutter", "partial_crop")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_boundary_scene(
    doc: np.ndarray, contrast: int, seed: int, sigma: float, background: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Create a flat desk/page scene with controlled paper-minus-desk contrast."""
    height, width = 900, 700
    source = doc if doc.ndim == 3 else cv2.cvtColor(doc, cv2.COLOR_GRAY2BGR)
    paper = np.where(source.astype(np.float32).mean(axis=2, keepdims=True) > 200,
                     background + contrast, source).astype(np.uint8)
    sh, sw = paper.shape[:2]
    target = np.array([[90, 70], [610, 70], [610, 830], [90, 830]], dtype=np.float32)
    origin = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(origin, target)
    warped = cv2.warpPerspective(paper, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
    occupied = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), transform, (width, height))
    scene = np.where(occupied[..., None] > 0, warped, background).astype(np.uint8)
    if sigma:
        random = np.random.default_rng(seed)
        scene = np.clip(scene.astype(np.float32) + random.normal(0, sigma, scene.shape), 0, 255).astype(np.uint8)
    return scene, target


def _run_brightness_control(doc: np.ndarray, out_dir: Path) -> None:
    """Hold geometry, contrast, source, seed and noise fixed; vary only base gray."""
    rows: list[dict[str, object]] = []
    for background in (100, 205):
        for method, strategy, mode in (
            ("canny_fixed", "canny", "fixed"), ("canny_auto", "canny", "auto"),
            ("canny_otsu", "canny", "otsu"), ("threshold", "threshold", "fixed"),
        ):
            for seed in SEEDS:
                scene, gt = _make_boundary_scene(doc, 10, seed, 0.0, background)
                result = detect_document(scene, strategy=strategy, canny_mode=mode)
                gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
                otsu_threshold, _ = cv2.threshold(
                    cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU,
                )
                error_px, error_ratio = corner_error(gt, result.corners)
                iou = quad_iou(gt, result.corners, scene.shape[:2])
                rows.append({
                    "ArkaPlan": background, "GriFark": 10, "Mod": method, "Seed": seed,
                    "Durum": result.status,
                    "Basarili": result.status == "ok" and error_ratio < 0.02 and iou > 0.90,
                    "KoseHata_oran": f"{error_ratio:.6f}", "IoU": f"{iou:.6f}",
                    "Skor": f"{result.score:.5f}", "Otsu_esigi": f"{otsu_threshold:.2f}",
                    "Neden": result.reason,
                })
    _write_csv(out_dir / "detection_contrast_brightness_trials.csv", rows)
    with (out_dir / "detection_contrast_brightness_trials.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    summary: list[dict[str, object]] = []
    for background in (100, 205):
        for method in ("canny_fixed", "canny_auto", "canny_otsu", "threshold"):
            group = [row for row in saved if row["ArkaPlan"] == str(background) and row["Mod"] == method]
            summary.append({"ArkaPlan": background, "GriFark": 10, "Mod": method,
                            "Basarili": sum(row["Basarili"] == "True" for row in group), "Toplam": len(group)})
    _write_csv(out_dir / "detection_contrast_brightness_summary.csv", summary)


def _run_contrast_sweep(doc: np.ndarray, out_dir: Path) -> None:
    raw: list[dict[str, object]] = []
    for contrast in CONTRASTS:
        for sigma in (0.0, 10.0):
            for strategy, mode in STRATEGIES:
                for seed in SEEDS:
                    scene, gt = _make_boundary_scene(doc, contrast, seed, sigma)
                    result = detect_document(scene, strategy=strategy, canny_mode=mode)
                    error_px, error_ratio = corner_error(gt, result.corners)
                    iou = quad_iou(gt, result.corners, scene.shape[:2])
                    passed = (
                        result.status == "ok" and error_ratio < 0.02 and iou > 0.90
                    )
                    raw.append({
                        "GriFark": contrast,
                        "Mod": f"{strategy}/{mode}",
                        "Noise_sigma": int(sigma),
                        "Seed": seed,
                        "Durum": result.status,
                        "Basarili": passed,
                        "KoseHata_px": f"{error_px:.3f}",
                        "KoseHata_oran": f"{error_ratio:.6f}",
                        "IoU": f"{iou:.6f}",
                        "Neden": result.reason,
                    })
    _write_csv(out_dir / "detection_contrast_trials.csv", raw)
    with (out_dir / "detection_contrast_trials.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in saved:
        key = (row["GriFark"], row["Mod"], row["Noise_sigma"])
        groups.setdefault(key, []).append(row)
    summary = [
        {
            "GriFark": key[0],
            "Mod": key[1],
            "Noise_sigma": key[2],
            "Basari_orani": f"{sum(row['Basarili'] == 'True' for row in values) / len(values):.5f}",
            "Basarili": sum(row["Basarili"] == "True" for row in values),
            "Toplam": len(values),
        }
        for key, values in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1], int(item[0][2])))
    ]
    _write_csv(out_dir / "detection_contrast_curve.csv", summary)


def _run_fallback(doc: np.ndarray, out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    overlay_dir = out_dir / "detection" / "fallback"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for scenario in FALLBACK_SCENARIOS:
        for seed in SEEDS:
            scene, gt = make_scene(doc, scenario, seed)
            result = process_image(scene, PipelineConfig(detection_fallback=True))
            error_px, error_ratio = corner_error(gt, result.detection.corners)
            iou = quad_iou(gt, result.detection.corners, scene.shape[:2])
            expected_reject = scenario == "partial_crop"
            correct = (
                result.detection.status != "ok"
                if expected_reject
                else result.detection.status == "ok" and error_ratio < 0.02 and iou > 0.90
            )
            fallback_step = next(
                (step["parametreler"] for step in result.steps if step["ad"] == "detect_fallback"),
                {},
            )
            rows.append({
                "Senaryo": scenario,
                "Seed": seed,
                "Beklenen": "reddet" if expected_reject else "tespit",
                "Birincil_durum": fallback_step.get("primary_status", result.detection.status),
                "Yedek_durum": fallback_step.get("fallback_status", "çalışmadı"),
                "Yedek_kabul": fallback_step.get("accepted", False),
                "Son_durum": result.detection.status,
                "KoseHata_px": f"{error_px:.3f}",
                "KoseHata_oran": f"{error_ratio:.6f}",
                "IoU": f"{iou:.6f}",
                "Dogru_sonuc": correct,
                "Neden": result.detection.reason,
            })
            save_image(overlay_dir / f"{scenario}_{seed}.png", _overlay(scene, gt, result.detection.corners))
    _write_csv(out_dir / "detection_fallback_trials.csv", rows)
    with (out_dir / "detection_fallback_trials.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in saved:
        groups.setdefault((row["Senaryo"], row["Beklenen"]), []).append(row)
    summary = [
        {
            "Senaryo": key[0],
            "Beklenen": key[1],
            "Dogru_sayi": sum(row["Dogru_sonuc"] == "True" for row in values),
            "Toplam": len(values),
            "Dogru_oran": f"{sum(row['Dogru_sonuc'] == 'True' for row in values) / len(values):.5f}",
        }
        for key, values in sorted(groups.items())
    ]
    _write_csv(out_dir / "detection_fallback_summary.csv", summary)


def run() -> None:
    output_dir = ROOT / "results"
    doc = _detection_document()
    _run_contrast_sweep(doc, output_dir)
    _run_fallback(doc, output_dir)
    _run_brightness_control(doc, output_dir)
    print(f"Gri fark eğrisi: {output_dir / 'detection_contrast_curve.csv'}")
    print(f"Fallback ölçümü: {output_dir / 'detection_fallback_summary.csv'}")


if __name__ == "__main__":
    run()
