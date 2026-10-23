"""Aşama 3 algılama deneylerini CSV ve overlay olarak üretir."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.image_info import save_image
from src.document.detection import detect_document
from src.document.perspective import four_point_transform, order_corners
from src.evaluation.metrics import corner_error, quad_iou, rectification_ssim
from src.evaluation.metrics import timed_median
from src.evaluation.scenes import SCENARIOS, make_scene


def _overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray | None) -> np.ndarray:
    result = image.copy()
    cv2.polylines(result, [np.rint(gt).astype(np.int32)], True, (0, 0, 255), 3)
    if pred is not None:
        cv2.polylines(result, [np.rint(pred).astype(np.int32)], True, (0, 255, 0), 3)
    return result


def _detection_document() -> np.ndarray:
    """Create a small in-page label with margin from every paper boundary."""
    doc = np.full((500, 360, 3), 245, np.uint8)
    cv2.putText(
        doc, "SMARTDOC", (100, 260), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (20, 20, 20), 2, cv2.LINE_AA,
    )
    return doc


def _boundary_measurements(
    gray: np.ndarray, edges: np.ndarray, gt: np.ndarray
) -> dict[str, float]:
    """Measure page-vs-desk contrast and Canny localization at the exterior edge.

    The inner strip is restricted to 1–3 pixels from the GT edge and excludes
    dark pixels, so text strokes cannot dominate the paper-side statistic.
    The desk strip is the corresponding 1–3 pixel exterior band.
    """
    page = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(page, [np.rint(gt).astype(np.int32)], 255)
    inside_distance = cv2.distanceTransform(page, cv2.DIST_L2, 3)
    outside_distance = cv2.distanceTransform(255 - page, cv2.DIST_L2, 3)
    paper_band = (page > 0) & (inside_distance <= 3.1) & (gray > 128)
    desk_band = (page == 0) & (outside_distance <= 3.1)
    paper_values = gray[paper_band]
    desk_values = gray[desk_band]
    paper_median = float(np.median(paper_values)) if paper_values.size else float("nan")
    desk_median = float(np.median(desk_values)) if desk_values.size else float("nan")
    delta = paper_median - desk_median

    full_edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    edge_pixels = full_edges > 0
    exterior_edge_band = (page == 0) & (outside_distance <= 3.1)
    edge_count = int(np.count_nonzero(edge_pixels))
    localized = int(np.count_nonzero(edge_pixels & exterior_edge_band))
    localized_ratio = localized / edge_count if edge_count else 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    exterior_edge_band = (page == 0) & (outside_distance <= 3.1)
    boundary_magnitude = magnitude[exterior_edge_band]
    background_texture = magnitude[(page == 0) & (outside_distance > 3.1)]
    text_pixels = ((page > 0) & (gray <= 128)).astype(np.uint8) * 255
    text_neighborhood = cv2.dilate(text_pixels, np.ones((7, 7), np.uint8)) > 0
    paper_texture_region = (page > 0) & (inside_distance > 3.1) & (gray > 128) & ~text_neighborhood
    paper_texture = magnitude[paper_texture_region]
    background_values = gray[(page == 0) & (outside_distance > 3.1)]
    paper_texture_values = gray[paper_texture_region]
    return {
        "paper_median": paper_median,
        "desk_median": desk_median,
        "boundary_delta": delta,
        "localized_ratio": localized_ratio,
        "localized_pixels": float(localized),
        "boundary_sobel_median": float(np.median(boundary_magnitude)) if boundary_magnitude.size else float("nan"),
        "boundary_sobel_p90": float(np.percentile(boundary_magnitude, 90)) if boundary_magnitude.size else float("nan"),
        "boundary_sobel_max": float(np.max(boundary_magnitude)) if boundary_magnitude.size else float("nan"),
        "background_sobel_p90": float(np.percentile(background_texture, 90)) if background_texture.size else float("nan"),
        "paper_sobel_p90": float(np.percentile(paper_texture, 90)) if paper_texture.size else float("nan"),
        "background_std": float(np.std(background_values)) if background_values.size else float("nan"),
        "paper_texture_std": float(np.std(paper_texture_values)) if paper_texture_values.size else float("nan"),
    }


def _corner_distances(gt: np.ndarray, pred: np.ndarray | None) -> tuple[list[float], float]:
    if pred is None:
        return [], float("nan")
    ordered_gt = order_corners(gt)
    ordered_pred = order_corners(pred)
    alignments = [np.roll(ordered_pred, shift, axis=0) for shift in range(4)]
    aligned_pred = min(
        alignments,
        key=lambda candidate: float(np.mean(np.linalg.norm(ordered_gt - candidate, axis=1))),
    )
    distances = np.linalg.norm(ordered_gt - aligned_pred, axis=1).astype(float).tolist()
    return distances, float(np.mean(distances))


def run() -> None:
    """Senaryo × strateji × Canny modu × seed ölçümlerini kaydeder."""
    result_dir = ROOT / "results" / "detection"
    result_dir.mkdir(parents=True, exist_ok=True)
    doc = _detection_document()
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for strategy in ("canny", "threshold"):
            modes = ("fixed", "auto", "otsu") if strategy == "canny" else ("fixed",)
            for mode in modes:
                for seed in range(5):
                    scene, gt = make_scene(doc, scenario, seed)
                    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
                    page_mask = np.zeros(scene.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(page_mask, [np.rint(gt).astype(np.int32)], 255)
                    paper_region = (page_mask > 0) & (gray_scene > 128)
                    paper_pixels = gray_scene[paper_region]
                    paper_mean = float(paper_pixels.mean()) if paper_pixels.size else float("nan")
                    background_pixels = gray_scene[page_mask == 0]
                    background_mean = float(background_pixels.mean()) if background_pixels.size else float("nan")
                    result, elapsed = timed_median(
                        detect_document,
                        scene,
                        strategy=strategy,
                        canny_mode=mode,
                        repeats=3,
                        warmup=1,
                    )
                    error_px, error_ratio = corner_error(gt, result.corners)
                    iou = quad_iou(gt, result.corners, scene.shape[:2])
                    expected = "reddet" if scenario == "partial_crop" else "tespit"
                    winner = "yok"
                    if result.corners is not None:
                        if scenario == "clutter":
                            clutter_corners = np.array([[20, 100], [680, 100], [680, 860], [20, 860]], dtype=np.float32)
                            clutter_iou = quad_iou(clutter_corners, result.corners, scene.shape[:2])
                            h, w = scene.shape[:2]
                            border_vertices = sum(
                                x <= 2 or y <= 2 or x >= w - 3 or y >= h - 3
                                for x, y in result.corners
                            )
                            if border_vertices >= 2:
                                winner = "arka_plan_siniri"
                            else:
                                winner = "clutter" if clutter_iou > iou else "belge"
                        else:
                            clutter_iou = float("nan")
                            winner = "belge" if iou > 0.90 else "yanlış_aday"
                    else:
                        clutter_iou = float("nan")
                    edge_map = result.debug["edges"]
                    edge_measurements = _boundary_measurements(gray_scene, edge_map, gt)
                    corner_distances, corner_distance_mean = _corner_distances(gt, result.corners)
                    ssim = float("nan")
                    if result.corners is not None:
                        try:
                            warped, _ = four_point_transform(scene, result.corners)
                            ssim = rectification_ssim(doc, warped)
                        except ValueError:
                            pass
                    save_image(result_dir / f"{scenario}_{strategy}_{mode}_{seed}.png", _overlay(scene, gt, result.corners))
                    rows.append({"Senaryo": scenario, "Strateji": strategy, "CannyMode": mode, "Seed": seed, "Noise_sigma": 10 if scenario == "low_contrast_hard_noisy" else 0, "Beklenen": expected, "Durum": result.status, "Skor": f"{result.score:.5f}", "KoseHata_px": f"{error_px:.3f}", "KoseHata_oran": f"{error_ratio:.5f}", "Kose_mesafeleri_px": ";".join(f"{value:.2f}" for value in corner_distances), "Kose_mesafesi_ortalama_px": f"{corner_distance_mean:.3f}", "IoU": f"{iou:.5f}", "KoseHata_esik": 0.02, "IoU_esik": 0.90, "SSIM_duzeltilmis": f"{ssim:.5f}", "Sure_ms_medyan": f"{elapsed:.3f}", "Neden": result.reason, "Kenar_piksel": result.debug["stages"]["edge_pixels"], "Closing_piksel": result.debug["stages"]["closing_pixels"], "Kontur": result.debug["stages"]["contours"], "Alan_gecen": result.debug["stages"]["area_pass"], "Hull": result.debug["stages"]["hulls"], "Approx_denemesi": result.debug["stages"]["approx_attempts"], "Approx_4_kose": result.debug["stages"]["approx_four"], "Skor_gecen": result.debug["stages"]["score_pass"], "Skor_kalan": result.debug["stages"]["score_reject"], "Gri_fark": f"{abs(paper_mean-background_mean):.2f}", "Kenar_kagıt_medyan": f"{edge_measurements['paper_median']:.2f}", "Kenar_masa_medyan": f"{edge_measurements['desk_median']:.2f}", "Kenar_fark_medyan": f"{edge_measurements['boundary_delta']:.2f}", "Kenar_Sobel_medyan": f"{edge_measurements['boundary_sobel_median']:.3f}", "Kenar_Sobel_p90": f"{edge_measurements['boundary_sobel_p90']:.3f}", "Kenar_Sobel_max": f"{edge_measurements['boundary_sobel_max']:.3f}", "Arka_plan_Sobel_p90": f"{edge_measurements['background_sobel_p90']:.3f}", "Kagit_doku_Sobel_p90": f"{edge_measurements['paper_sobel_p90']:.3f}", "Arka_plan_std": f"{edge_measurements['background_std']:.3f}", "Kagit_doku_std": f"{edge_measurements['paper_texture_std']:.3f}", "GT_dis_kenar_pikseli": int(edge_measurements["localized_pixels"]), "GT_dis_kenar_orani": f"{edge_measurements['localized_ratio']:.5f}", "Aday_GT": "evet" if result.corners is not None and iou > 0.90 else "hayir", "Aday_sahibi": winner, "Clutter_IoU": f"{clutter_iou:.5f}"})
    csv_path = ROOT / "results" / "detection_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Özet de aynı CSV'den kodla üretilir; başarısız satırlar korunur.
    summary: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        summary.setdefault((str(row["Senaryo"]), str(row["Strateji"]), str(row["CannyMode"])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for key, values in sorted(summary.items()):
        expected = str(values[0]["Beklenen"])
        if expected == "tespit":
            correct = sum(
                v["Durum"] == "ok"
                and float(v["KoseHata_oran"]) < float(v["KoseHata_esik"])
                and float(v["IoU"]) > float(v["IoU_esik"])
                for v in values
            )
            rate_key = "Basari_orani"
        else:
            correct = sum(v["Durum"] != "ok" or v["Aday_sahibi"] == "yok" for v in values)
            rate_key = "Dogru_reddetme_orani"
        summary_rows.append({"Senaryo": key[0], "Strateji": key[1], "CannyMode": key[2], "Beklenen": expected, "Dogru_sayi": correct, "Toplam": len(values), rate_key: f"{correct / len(values):.3f}"})
        print(*key, f"beklenen={expected}", f"{rate_key}={correct / len(values):.3f}")
    with (ROOT / "results" / "detection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("Senaryo", "Strateji", "CannyMode", "Beklenen", "Dogru_sayi", "Toplam", "Basari_orani", "Dogru_reddetme_orani")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in summary_rows)
    diagnostic_cases = {"rotated_30", "rotated_45", "rotated_90", "strong_perspective", "clutter", "low_contrast_bg", "low_contrast_hard"}
    stage_fields = ("Kenar_piksel", "Closing_piksel", "Kontur", "Alan_gecen", "Hull", "Approx_denemesi", "Approx_4_kose", "Skor_gecen", "Skor_kalan")
    with (ROOT / "results" / "detection_stage_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Senaryo", "Strateji", "CannyMode", *stage_fields))
        writer.writeheader()
        for key, values in sorted(summary.items()):
            if key[0] not in diagnostic_cases:
                continue
            record: dict[str, object] = dict(zip(("Senaryo", "Strateji", "CannyMode"), key))
            for field in stage_fields:
                record[field] = round(sum(float(value[field]) for value in values) / len(values), 2)
            writer.writerow(record)

    edge_diagnostics = [
        row for row in rows
        if row["Senaryo"] in {"low_contrast_bg", "low_contrast_hard", "low_contrast_hard_noisy"}
        and row["Strateji"] == "canny"
        and row["CannyMode"] == "fixed"
    ]
    edge_summary_path = ROOT / "results" / "detection_edge_summary.csv"
    with edge_summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("Senaryo", "Gri_fark", "Kenar_kagıt_medyan", "Kenar_masa_medyan", "Kenar_fark_medyan", "Sobel_medyan", "Sobel_p90", "Sobel_max", "Arka_plan_Sobel_p90", "Kagit_doku_Sobel_p90", "Arka_plan_std", "Kagit_doku_std", "GT_dis_kenar_orani_ortalama", "Aday_GT_sayisi", "Toplam")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in ("low_contrast_bg", "low_contrast_hard", "low_contrast_hard_noisy"):
            group = [row for row in edge_diagnostics if row["Senaryo"] == scenario]
            if not group:
                continue
            writer.writerow({
                "Senaryo": scenario,
                "Gri_fark": group[0]["Gri_fark"],
                "Kenar_kagıt_medyan": group[0]["Kenar_kagıt_medyan"],
                "Kenar_masa_medyan": group[0]["Kenar_masa_medyan"],
                "Kenar_fark_medyan": group[0]["Kenar_fark_medyan"],
                "Sobel_medyan": f"{np.median([float(row['Kenar_Sobel_medyan']) for row in group]):.3f}",
                "Sobel_p90": f"{np.median([float(row['Kenar_Sobel_p90']) for row in group]):.3f}",
                "Sobel_max": f"{np.median([float(row['Kenar_Sobel_max']) for row in group]):.3f}",
                "Arka_plan_Sobel_p90": f"{np.median([float(row['Arka_plan_Sobel_p90']) for row in group]):.3f}",
                "Kagit_doku_Sobel_p90": f"{np.median([float(row['Kagit_doku_Sobel_p90']) for row in group]):.3f}",
                "Arka_plan_std": f"{np.median([float(row['Arka_plan_std']) for row in group]):.3f}",
                "Kagit_doku_std": f"{np.median([float(row['Kagit_doku_std']) for row in group]):.3f}",
                "GT_dis_kenar_orani_ortalama": f"{np.mean([float(row['GT_dis_kenar_orani']) for row in group]):.5f}",
                "Aday_GT_sayisi": sum(row["Aday_GT"] == "evet" for row in group),
                "Toplam": len(group),
            })

    # Record both construction-level edge enhancers and measurements from scenes.
    low_contrast_scenes = {"low_contrast_bg", "low_contrast_hard", "low_contrast_hard_noisy", "clutter"}
    inventory_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        group = [
            row for row in rows
            if row["Senaryo"] == scenario and row["Strateji"] == "canny" and row["CannyMode"] == "fixed"
        ]
        if not group:
            continue
        if scenario in low_contrast_scenes:
            background_description = "düz masa 205; clutter'da ek dikdörtgen"
            background_noise_sigma = 0
            background_gradient_range = 0
        elif scenario == "frontal_clean":
            background_description = "düz masa 120"
            background_noise_sigma = 0
            background_gradient_range = 0
        elif scenario == "dark_bg":
            background_description = "düz zemin 35"
            background_noise_sigma = 0
            background_gradient_range = 0
        else:
            background_description = "100→155 doğrusal gradyan + Gaussian doku"
            background_noise_sigma = 8
            background_gradient_range = 55
        inventory_rows.append({
            "Senaryo": scenario,
            "Arka_plan_kurulumu": background_description,
            "Arka_plan_gradyan_araligi": background_gradient_range,
            "Arka_plan_gurultu_sigma": background_noise_sigma,
            "Kagit_doku_sigma": 0,
            "Ek_kagit_golgesi_intensitesi": 30 if scenario == "noisy_shadow" else 0,
            "Ek_sensor_gurultu_sigma": 10 if scenario == "low_contrast_hard_noisy" else 0,
            "Ek_outline_kalinligi_px": 5 if scenario == "clutter" else 0,
            "Occupancy_sinir_katmani": "sert ikili kompozit",
            "Olculen_arka_plan_std": f"{np.median([float(row['Arka_plan_std']) for row in group]):.3f}",
            "Olculen_arka_plan_Sobel_p90": f"{np.median([float(row['Arka_plan_Sobel_p90']) for row in group]):.3f}",
            "Olculen_kagit_Sobel_p90": f"{np.median([float(row['Kagit_doku_Sobel_p90']) for row in group]):.3f}",
            "Sinir_paper_minus_desk_medyan": group[0]["Kenar_fark_medyan"],
            "GT_dis_band_Canny_orani": f"{np.median([float(row['GT_dis_kenar_orani']) for row in group]):.5f}",
            "Secilen_kose_hatasi_px": f"{np.median([float(row['Kose_mesafesi_ortalama_px']) for row in group]):.3f}",
            "Belge_adayi_sayisi": sum(row["Aday_sahibi"] == "belge" for row in group),
            "Toplam": len(group),
        })
    with (ROOT / "results" / "scene_edge_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    polarity = np.full((420, 320), 235, dtype=np.uint8)
    polarity_gt = np.array([[35, 45], [285, 45], [285, 375], [35, 375]], dtype=np.float32)
    cv2.fillPoly(polarity, [polarity_gt.astype(np.int32)], 38)
    polarity_result = detect_document(polarity, strategy="threshold")
    polarity_iou = quad_iou(polarity_gt, polarity_result.corners, polarity.shape)
    if polarity_result.corners is None:
        polarity_owner = "aday_yok"
    elif polarity_iou > 0.90:
        polarity_owner = "belge"
    else:
        polarity_owner = "yanlis_aday"
    polarity_overlay = cv2.cvtColor(polarity, cv2.COLOR_GRAY2BGR)
    cv2.polylines(polarity_overlay, [polarity_gt.astype(np.int32)], True, (0, 0, 255), 2)
    if polarity_result.corners is not None:
        cv2.polylines(polarity_overlay, [np.rint(polarity_result.corners).astype(np.int32)], True, (0, 255, 0), 2)
    save_image(ROOT / "results" / "detection" / "polarity_dark_paper_threshold.png", polarity_overlay)
    with (ROOT / "results" / "polarity_observation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("Strateji", "Durum", "Aday_sahibi", "IoU", "GT_koseleri", "Tahmin_koseleri", "Neden")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Strateji": "threshold",
            "Durum": polarity_result.status,
            "Aday_sahibi": polarity_owner,
            "IoU": f"{polarity_iou:.5f}",
            "GT_koseleri": np.array2string(polarity_gt, precision=1),
            "Tahmin_koseleri": np.array2string(polarity_result.corners, precision=1) if polarity_result.corners is not None else "None",
            "Neden": polarity_result.reason,
        })
    print(f"Ters polarite threshold: status={polarity_result.status}; Aday_sahibi={polarity_owner}; IoU={polarity_iou:.5f}")
    print(f"GT köşeleri: {polarity_gt.tolist()}")
    print(f"Tahmin köşeleri: {polarity_result.corners.tolist() if polarity_result.corners is not None else None}")


if __name__ == "__main__":
    run()
