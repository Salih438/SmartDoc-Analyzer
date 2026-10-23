"""Sprint 4B doğrulama ölçümlerini tekrar üretir."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_sauvola

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import f1_score
from src.evaluation.scenes import make_scene
from src.export.pdf import _page_canvas, _to_pdf_image
from src.document.perspective import compute_output_size
from src.pipeline import PipelineConfig, process_image
from src.preprocessing.illumination import _scaled_odd_ksize, flatten_background
from src.scan import SCAN_MODES, make_scan


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size)


def measure_flatten() -> tuple[float, int]:
    height, width = 3000, 4000
    rng = np.random.default_rng(411)
    yy, xx = np.mgrid[:height, :width]
    illumination = 150 + 50 * xx / width + 20 * yy / height
    texture = rng.normal(0, 5, (height, width))
    gray = np.clip(illumination + texture, 0, 255).astype(np.uint8)
    gray[::31, :] = np.clip(gray[::31, :].astype(np.int16) - 15, 0, 255).astype(np.uint8)
    reduced = flatten_background(gray)
    full_kernel = _scaled_odd_ksize(None, gray.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (full_kernel, full_kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    full = cv2.divide(gray, bg, scale=255)
    difference = cv2.absdiff(reduced, full)
    result = np.concatenate((reduced, full), axis=1)
    target = ROOT / "results" / "flatten_compare.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), result)
    return float(difference.mean()), int(difference.max())


def measure_edges() -> list[tuple[str, str, float, float, float, float]]:
    doc = np.full((560, 430, 3), 248, np.uint8)
    cv2.putText(doc, "SMARTDOC", (35, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(doc, "Belge test metni", (35, 280), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2, cv2.LINE_AA)
    rows = []
    for scenario, seed in (("frontal_easy", 401), ("noisy_shadow", 402), ("strong_perspective", 403)):
        scene, gt_corners = make_scene(doc, scenario, seed)
        result = process_image(scene, PipelineConfig(binarize="none"))
        source = result.rectified_image if result.rectified_image is not None else scene
        if scenario == "strong_perspective" and result.detection.corners is not None:
            corners = result.detection.corners.astype(np.float32)
            out_w, out_h = compute_output_size(corners)
            destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
            matrix = cv2.getPerspectiveTransform(corners, destination)
            replicated = cv2.warpPerspective(scene, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
            compare = np.concatenate((source, replicated), axis=1)
            cv2.imwrite(str(ROOT / "results" / "edge_asymmetry.png"), compare)
            def side_delta(arr: np.ndarray) -> float:
                gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                return float(np.mean(gray_arr[:, :5]) - np.mean(gray_arr[:, -5:]))
            print("A3 strong_perspective left-right, default/replicate:", side_delta(source), side_delta(replicated))
            gt_w, gt_h = compute_output_size(gt_corners)
            gt_destination = np.array([[0, 0], [gt_w - 1, 0], [gt_w - 1, gt_h - 1], [0, gt_h - 1]], np.float32)
            gt_matrix = cv2.getPerspectiveTransform(gt_corners.astype(np.float32), gt_destination)
            gt_warp = cv2.warpPerspective(scene, gt_matrix, (gt_w, gt_h), borderMode=cv2.BORDER_REPLICATE)
            print("A3 strong_perspective GT-warp left-right:", side_delta(gt_warp))
        for mode in SCAN_MODES:
            out = make_scan(source, mode, inset_fraction=0)
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
            strip = max(1, round(min(gray.shape) * .01))
            sides = [gray[:5], gray[-5:], gray[:, :5], gray[:, -5:]]
            values = [float(np.mean(s)) for s in sides]
            inner = float(np.median(gray[strip:-strip, strip:-strip]))
            rows.append((scenario, mode, *values, inner))
    return rows


def measure_sauvola() -> list[tuple[str, int, float, float]]:
    width, height = 1100, 1450
    page = Image.new("L", (width, height), 255)
    ink = Image.new("L", (width, height), 0)
    draw, gt = ImageDraw.Draw(page), ImageDraw.Draw(ink)
    draw.text((90, 90), "BOLD DOCUMENT TITLE", font=_font(58), fill=0)
    gt.text((90, 90), "BOLD DOCUMENT TITLE", font=_font(58), fill=255)
    y = 240
    for line in ("Small body text line one", "Small body text line two", "Thin rule below"):
        draw.text((90, y), line, font=_font(27), fill=0)
        gt.text((90, y), line, font=_font(27), fill=255)
        y += 95
    draw.line((90, y + 20, 850, y + 20), fill=0, width=1)
    gt.line((90, y + 20, 850, y + 20), fill=255, width=1)
    gray = np.asarray(page)
    truth = np.asarray(ink) > 0
    line_region = np.zeros_like(truth)
    line_region[y + 20, 90:851] = True
    rows = []
    for label, region in (("title", (slice(70, 190), slice(70, 950))), ("body", (slice(220, 600), slice(70, 950)))):
        for window in (31, 61, 91, 99):
            threshold = threshold_sauvola(gray, window_size=window, k=.2)
            pred = gray <= threshold
            region_mask = np.zeros_like(truth)
            region_mask[region] = True
            gt_crop = (truth & region_mask).astype(np.uint8)
            pred_crop = (pred & region_mask).astype(np.uint8)
            f1 = f1_score(gt_crop, pred_crop)
            line_retention = float(np.count_nonzero(pred & truth & line_region) / max(1, np.count_nonzero(truth & line_region)))
            rows.append((label, window, f1, line_retention))
    return rows


def measure_pdf_lines() -> tuple[float, float]:
    source = np.full((4000, 3000), 255, np.uint8)
    for x in range(120, 2880, 90):
        source[300:3700, x] = 0
    binary = _to_pdf_image(source, 1, lossless_binary=True)
    nearest_scaled = binary.resize((2480, 3307), Image.Resampling.NEAREST)
    nearest = Image.new("1", (2480, 3508), 1)
    nearest.paste(nearest_scaled, (0, (3508 - nearest_scaled.height) // 2))
    gray = Image.fromarray(source).resize((2480, 3307), Image.Resampling.LANCZOS)
    lanczos = Image.fromarray(np.where(np.asarray(gray) < 128, 0, 255).astype(np.uint8)).convert("1")
    # Same A4 canvas placement as _page_canvas, retaining a binary comparison page.
    canvas = Image.new("1", (2480, 3508), 1)
    canvas.paste(lanczos, (0, (3508 - lanczos.height) // 2))
    nearest_arr = np.asarray(nearest.convert("L"))
    lanczos_arr = np.asarray(canvas.convert("L"))
    positions = [round(x * (3307 / 4000)) for x in range(120, 2880, 90)]
    def retained(arr: np.ndarray) -> float:
        found = sum(bool(np.any(arr[120:3200, x:x + 1] < 128)) for x in positions)
        return found / len(positions)
    return retained(nearest_arr), retained(lanczos_arr)


def main() -> None:
    print("A2 flatten 4000x3000 mean_abs,max_abs:", measure_flatten())
    print("A3 scenario,mode,top5,bottom5,left5,right5,inner_median")
    for row in measure_edges():
        print(*row, sep=",")
    print("A4 group,window,F1,line_retention")
    for row in measure_sauvola():
        print(*row, sep=",")
    print("A5 line retention (NEAREST,LANCZOS_threshold):", measure_pdf_lines())


if __name__ == "__main__":
    main()
