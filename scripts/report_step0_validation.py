"""Produce ADIM 0 audit tables from the saved detection CSV measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_detection_experiments import _detection_document
from src.document.detection import detect_document
from src.evaluation.metrics import corner_error, quad_iou


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Ölçüm satırı üretilemedi: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _category(row: dict[str, str], expected: str) -> str:
    status = row.get("Durum", row.get("Son_durum", "not_found"))
    is_correct = False
    if expected == "reddet":
        is_correct = status != "ok"
    else:
        try:
            is_correct = (
                status == "ok"
                and float(row.get("KoseHata_oran", "nan")) < 0.02
                and float(row.get("IoU", "nan")) > 0.90
            )
        except ValueError:
            is_correct = False
    if status == "ok":
        return "ok-doğru" if is_correct else "ok-YANLIŞ"
    return status if status in {"low_confidence", "not_found"} else "not_found"


def _confusion(source: str, rows: list[dict[str, str]], method_key, expected_key="Beklenen") -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(method_key(row)), []).append(row)
    categories = ("ok-doğru", "ok-YANLIŞ", "low_confidence", "not_found")
    summary = []
    for method, values in sorted(groups.items()):
        expected = values[0].get(expected_key, "tespit")
        counts = {category: sum(_category(row, expected) == category for row in values) for category in categories}
        summary.append({"Kaynak": source, "Yöntem": method, **counts, "N": len(values)})
    return summary


def _fallback_area_sweep(doc: np.ndarray) -> list[dict[str, object]]:
    height, width = 900, 700
    target_aspect = width / height
    rows: list[dict[str, object]] = []
    for requested_area in np.arange(0.40, 0.951, 0.05):
        page_h = int(round(np.sqrt(requested_area * height * width / target_aspect)))
        page_w = int(round(page_h * target_aspect))
        x0, y0 = (width - page_w) // 2, (height - page_h) // 2
        corners = np.array([[x0, y0], [x0 + page_w - 1, y0],
                            [x0 + page_w - 1, y0 + page_h - 1], [x0, y0 + page_h - 1]], np.float32)
        source = cv2.resize(doc, (page_w, page_h), interpolation=cv2.INTER_AREA)
        scene = np.full((height, width, 3), 100, np.uint8)
        scene[y0:y0 + page_h, x0:x0 + page_w] = source
        detection = detect_document(scene, strategy="threshold")
        actual_area = 0.0
        touches_frame = False
        error_ratio, overlap = 1.0, 0.0
        if detection.corners is not None:
            actual_area = abs(float(cv2.contourArea(detection.corners))) / (height * width)
            error_px, error_ratio = corner_error(corners, detection.corners)
            overlap = quad_iou(corners, detection.corners, scene.shape[:2])
            margin = max(2.0, min(height, width) * 0.01)
            touches_frame = bool(
                np.any(detection.corners[:, 0] <= margin) or np.any(detection.corners[:, 1] <= margin)
                or np.any(detection.corners[:, 0] >= width - 1 - margin)
                or np.any(detection.corners[:, 1] >= height - 1 - margin)
            )
        candidate_valid = detection.corners is not None and error_ratio < 0.02 and overlap > 0.90
        gate_accepts = candidate_valid and not touches_frame and actual_area <= 0.75
        rows.append({
            "Alan_hedef": f"{requested_area:.2f}", "Alan_gercek": f"{actual_area:.5f}",
            "Durum": detection.status, "Dogru_aday": candidate_valid,
            "Sinira_yakin": touches_frame, "Fallback_75_kabul": gate_accepts,
            "Yanlis_ret": candidate_valid and not gate_accepts,
            "Yanlis_ret_alan_kapisi": candidate_valid and not touches_frame and actual_area > 0.75,
            "KoseHata_oran": f"{error_ratio:.6f}", "IoU": f"{overlap:.6f}",
            "Skor": f"{detection.score:.6f}",
        })
    return rows


def run() -> None:
    results = ROOT / "results"
    detection = _read(results / "detection_metrics.csv")
    contrast = _read(results / "detection_contrast_trials.csv")
    fallback = _read(results / "detection_fallback_trials.csv")
    median = _read(results / "pipeline_detection_denoise.csv")
    all_rows = []
    all_rows += _confusion("ana", detection, lambda r: f"{r['Strateji']}/{r['CannyMode']}")
    all_rows += _confusion("sensitivity", contrast, lambda r: f"{r['Mod']}/noise_sigma={r['Noise_sigma']}")
    all_rows += _confusion("fallback", fallback, lambda r: "canny/fixed->threshold", "Beklenen")
    all_rows += _confusion("median", median, lambda r: f"{r['Senaryo']}/{r['Yontem']}", "Beklenen")
    _write(results / "detection_confusion.csv", all_rows)

    doc = _detection_document()
    area_rows = _fallback_area_sweep(doc)
    _write(results / "fallback_area_sweep.csv", area_rows)
    eligible = [row for row in area_rows if row["Dogru_aday"] and not row["Sinira_yakin"]]
    area_rejections = sum(bool(row["Yanlis_ret_alan_kapisi"]) for row in eligible)
    area_summary = [{
        "Alan_siniri": 0.75, "Temiz_tek_belge_dogru_aday_sinir_ici": len(eligible),
        "Alan_nedeniyle_yanlis_ret": area_rejections,
        "Yanlis_ret_orani": f"{area_rejections / max(1, len(eligible)):.6f}",
        "Alan_sinirini_asip_yaklasmayan_aday": sum(
            bool(row["Dogru_aday"] and float(row["Alan_gercek"]) > 0.75 and not row["Sinira_yakin"])
            for row in area_rows
        ),
    }]
    _write(results / "fallback_area_summary.csv", area_summary)

    score_rows = []
    for row in detection:
        if row["Senaryo"] in {"clutter", "clutter_highcontrast"}:
            score_rows.append({"Senaryo": row["Senaryo"], "Yöntem": row["Strateji"], "Seed": row["Seed"],
                               "Durum": row["Durum"], "Skor": row["Skor"],
                               "Skor_bilesenleri": row.get("Skor_bilesenleri", ""), "Aday_sahibi": row["Aday_sahibi"],
                               "Belge_IoU": row["IoU"], "KoseHata_oran": row["KoseHata_oran"],
                               "Clutter_IoU": row["Clutter_IoU"]})
    for row in median:
        if row["Senaryo"] == "noisy_shadow" and row["Noise_sigma"] == "40.0" and row["Yontem"] == "raw":
            score_rows.append({"Senaryo": row["Senaryo"], "Yöntem": "canny/fixed raw σ40", "Seed": row["Seed"],
                               "Durum": row["Durum"], "Skor": row.get("Skor", ""),
                               "Skor_bilesenleri": row.get("Skor_bilesenleri", ""), "Aday_sahibi": "",
                               "Belge_IoU": row["IoU"], "KoseHata_oran": row["KoseHata_oran"], "Clutter_IoU": ""})
    _write(results / "detection_score_samples.csv", score_rows)

    validation_rows: list[dict[str, object]] = []
    for row in score_rows:
        components = json.loads(str(row["Skor_bilesenleri"] or "{}"))
        score = float(row["Skor"] or 0.0)
        # Reconstruct the previously used MIN_SCORE=.45 status from the saved score.
        old_status = "not_found" if not components else ("ok" if score >= 0.45 else "low_confidence")
        validation_rows.append({
            "Senaryo": row["Senaryo"], "Yöntem": row["Yöntem"], "Seed": row["Seed"],
            "Durum_once_MIN_SCORE_0_45": old_status,
            "Durum_sonra_MIN_SCORE_0_67_area_guard": row["Durum"],
            "Skor_sonra": row["Skor"], "KoseHata_oran_sonra": row.get("KoseHata_oran", ""),
            "IoU_sonra": row.get("Belge_IoU", ""), "Aday_sahibi": row.get("Aday_sahibi", ""),
        })
    _write(results / "detection_score_before_after.csv", validation_rows)

    brightness = _read(results / "detection_contrast_brightness_summary.csv")
    print("Durum-dogruluk matrisi (detection_confusion.csv):")
    for row in all_rows:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))
    print("Temiz tek-belge alan taraması (fallback_area_sweep.csv):")
    print("Alan kapısı özeti (fallback_area_summary.csv):")
    for row in area_summary:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))
    for row in area_rows:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))
    print("Tek değişkenli parlaklık kontrolü:")
    for row in brightness:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))
    print(f"Skor örnekleri: results/detection_score_samples.csv ({len(score_rows)} satır)")


if __name__ == "__main__":
    run()
