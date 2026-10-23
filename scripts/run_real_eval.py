"""Evaluate optional manually annotated photos in data/test."""

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

from src.analysis.image_info import load_image, save_image
from src.document.detection import detect_document
from src.document.perspective import four_point_transform
from src.evaluation.metrics import corner_error, quad_iou
from src.evaluation.text_metrics import cer, wer
from src.ocr.engine import is_available, run_ocr


def run() -> None:
    data_dir = ROOT / "data" / "test"
    output = ROOT / "results" / "real_eval.csv"
    rows: list[dict[str, object]] = []
    images = sorted(path for path in data_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
    if not images:
        print("data/test icinde gercek fotograf yok.")
    overlay_dir = ROOT / "results" / "real_eval"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        image = load_image(path)
        annotation = path.with_suffix(".corners.json")
        if not annotation.is_file():
            rows.append({"Dosya": path.name, "Durum": "ground_truth_yok", "KoseHata_px": "", "IoU": "", "CER": "", "WER": ""})
            continue
        gt = np.asarray(json.loads(annotation.read_text(encoding="utf-8"))["corners"], dtype=np.float32)
        found = detect_document(image)
        err_px, err_ratio = corner_error(gt, found.corners)
        overlap = quad_iou(gt, found.corners, image.shape[:2])
        overlay = image.copy()
        cv2.polylines(overlay, [np.rint(gt).astype(np.int32)], True, (0, 0, 255), 3)
        if found.corners is not None:
            cv2.polylines(overlay, [np.rint(found.corners).astype(np.int32)], True, (0, 255, 0), 3)
        save_image(overlay_dir / f"{path.stem}_overlay.png", overlay)
        text_path = path.with_suffix(".txt")
        predicted_text = ""
        status = found.status
        if found.status == "ok" and found.corners is not None:
            warped, _ = four_point_transform(image, found.corners)
            save_image(overlay_dir / f"{path.stem}_rectified.png", warped)
            if text_path.is_file() and is_available():
                try:
                    predicted_text = run_ocr(warped).text
                except RuntimeError as exc:
                    status = f"ocr_hatası:{exc}"
        ground_text = text_path.read_text(encoding="utf-8") if text_path.is_file() else None
        rows.append({"Dosya": path.name, "Durum": status, "KoseHata_px": f"{err_px:.3f}",
                     "KoseHata_oran": f"{err_ratio:.6f}", "IoU": f"{overlap:.6f}",
                     "CER": f"{cer(ground_text, predicted_text):.6f}" if ground_text is not None and predicted_text else "",
                     "WER": f"{wer(ground_text, predicted_text):.6f}" if ground_text is not None and predicted_text else ""})
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["Dosya", "Durum", "KoseHata_px", "KoseHata_oran", "IoU", "CER", "WER"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Gercek fotograf degerlendirmesi: {len(rows)} kayit; CSV={output}")


if __name__ == "__main__":
    run()
