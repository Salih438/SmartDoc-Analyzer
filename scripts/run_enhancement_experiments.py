"""Sabit seed'lerle belge iyileştirme ve eşikleme ölçümlerini CSV'ye yazar."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_niblack, threshold_sauvola

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.comparison import compare_enhancements
from src.evaluation.metrics import f1_score, iou_score, psnr, sharpness_laplacian_var, ssim
from src.evaluation.noise import simulate_uneven_illumination
from src.evaluation.ocr_document import _font_path
from src.preprocessing.enhancement import apply_clahe, document_stretch, gamma_correction
from src.preprocessing.illumination import flatten_background
from src.segmentation.threshold import adaptive_threshold, otsu_threshold


SEEDS = (11, 23, 37, 51, 69)
CONDITIONS = ("karanlik", "dusuk_kontrast", "dengesiz_aydinlatma", "asiri_parlak")
BASE_METHOD_NAMES = (
    "none", "contrast_stretch", "gamma_auto", "gamma_0.6", "gamma_0.8",
    "gamma_1.2", "gamma_1.6", "gamma_2.0", "clahe_1.0", "clahe_2.0",
    "clahe_4.0", "histogram_equalization",
)
EXTRA_GRAY_METHOD_NAMES = (
    "gamma_2.5", "gamma_3.0", "gamma_paper", "document_stretch",
)
THRESHOLD_METHOD_NAMES = (
    "otsu", "adaptive_gaussian", "adaptive_mean", "sauvola", "niblack",
    "flatten+otsu", "flatten+clahe+otsu",
)
METHOD_NAMES = BASE_METHOD_NAMES + EXTRA_GRAY_METHOD_NAMES + THRESHOLD_METHOD_NAMES
GRAY_METRICS = ("PSNR", "SSIM", "Kontrast_std", "Keskinlik_laplacian")
ALL_METRICS = GRAY_METRICS + ("F1", "IoU")
DOC_SPECS = {
    "belge_a": (25, 1.00, ["SMARTDOC ANALYZER", "ÇAĞRI İSTASYONU", "çağrı", "Işık ve gölge ölçümü"]),
    "belge_b": (30, 1.22, ["Belge Deney Raporu", "Ölçüm: 42,7 mm", "Şube: Çankaya", "Tarih: 23.10.2026"]),
    "belge_c": (22, 0.82, ["ARŞİV KAYDI", "İsim: Çağrı Şahin", "Bölüm: Görüntü İşleme", "Not: ğ ü ö ş ı İ ç"]),
}


def _condition_input(clean: np.ndarray, condition: str, seed: int) -> np.ndarray:
    """Temiz gri belgeye tekrarlanabilir ışık/kontrast dönüşümü uygular."""
    rng = np.random.default_rng(seed)
    if condition == "karanlik":
        gamma = float(rng.uniform(0.48, 0.62))
        exposure = float(rng.uniform(0.52, 0.68))
        adjusted = gamma_correction(clean, gamma)
        return np.clip(adjusted.astype(np.float32) * exposure, 0, 255).astype(np.uint8)
    if condition == "dusuk_kontrast":
        alpha = float(rng.uniform(0.15, 0.22))
        transformed = (clean.astype(np.float32) - 127.5) * alpha + 127.5
        return np.clip(np.rint(transformed), 0, 255).astype(np.uint8)
    if condition == "dengesiz_aydinlatma":
        return simulate_uneven_illumination(clean, strength=0.7, seed=seed)
    if condition == "asiri_parlak":
        gamma = float(rng.uniform(1.8, 2.2))
        return gamma_correction(clean, gamma)
    raise ValueError(f"Bilinmeyen deney koşulu: {condition}")


def _make_document(doc_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Farklı metin, puntolar ve satır aralıklarıyla BGR belge ve GT maskesi üretir."""
    if doc_id not in DOC_SPECS:
        raise ValueError(f"Bilinmeyen belge kimliği: {doc_id}")
    font_size, spacing_scale, lines = DOC_SPECS[doc_id]
    font = ImageFont.truetype(str(_font_path()), size=font_size)
    page = Image.new("L", (700, 900), 255)
    mask = Image.new("L", page.size, 0)
    page_draw, mask_draw = ImageDraw.Draw(page), ImageDraw.Draw(mask)
    y = 140
    for line in lines:
        xy = (60, y)
        page_draw.text(xy, line, font=font, fill=25)
        mask_draw.text(xy, line, font=font, fill=255)
        y += round(font_size * 2.6 * spacing_scale)
    page_array = np.asarray(page, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    binary_mask = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    return cv2.cvtColor(page_array, cv2.COLOR_GRAY2BGR), binary_mask


def _gamma_paper(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """90. yüzdelik kâğıt seviyesini yaklaşık 0.9*255'e taşıyacak gamma'yı bulur."""
    paper_level = float(np.percentile(gray, 90))
    target = 0.9 * 255.0
    if paper_level <= 0.0 or paper_level >= 255.0 or target >= 255.0:
        gamma = 1.0
    else:
        gamma = float(np.log(paper_level / 255.0) / np.log(target / 255.0))
        gamma = float(np.clip(gamma, 0.1, 5.0))
    return gamma_correction(gray, gamma), gamma


def _all_outputs(gray: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Gri iyileştirme kollarını ve yedi eşikleme kolunu üretir."""
    enhanced = {name: item["image"] for name, item in compare_enhancements(gray).items()}
    enhanced["gamma_2.5"] = gamma_correction(gray, 2.5)
    enhanced["gamma_3.0"] = gamma_correction(gray, 3.0)
    enhanced["gamma_paper"], _ = _gamma_paper(gray)
    enhanced["document_stretch"] = document_stretch(gray)

    binary: dict[str, np.ndarray] = {}
    for name, image in enhanced.items():
        binary[name], _ = otsu_threshold(image, invert=True)

    binary["otsu"], _ = otsu_threshold(gray, invert=True)
    binary["adaptive_gaussian"] = adaptive_threshold(
        gray, method="gaussian", block_size=31, c=5, invert=True
    )
    binary["adaptive_mean"] = adaptive_threshold(
        gray, method="mean", block_size=31, c=5, invert=True
    )
    sauvola_threshold_map = threshold_sauvola(gray, window_size=31, k=0.2)
    binary["sauvola"] = np.where(gray < sauvola_threshold_map, 255, 0).astype(np.uint8)
    niblack_threshold_map = threshold_niblack(gray, window_size=31, k=-0.2)
    binary["niblack"] = np.where(gray < niblack_threshold_map, 255, 0).astype(np.uint8)
    flattened = flatten_background(gray, ksize=301)
    binary["flatten+otsu"], _ = otsu_threshold(flattened, invert=True)
    flattened_clahe = apply_clahe(flattened, clip_limit=2.0)
    binary["flatten+clahe+otsu"], _ = otsu_threshold(flattened_clahe, invert=True)
    return enhanced, binary


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _make_montage(
    outputs_by_doc: dict[str, dict[str, np.ndarray]], condition: str, output_path: Path
) -> None:
    """Bir koşulun belge×yöntem ikili çıktılarından etiketli küçük montaj üretir."""
    tile_w, tile_h, columns = 190, 230, 6
    methods_per_doc = list(next(iter(outputs_by_doc.values())))
    total_rows = len(outputs_by_doc) * int(np.ceil(len(methods_per_doc) / columns))
    montage = np.full((total_rows * tile_h, columns * tile_w), 245, dtype=np.uint8)
    for doc_index, (doc_id, outputs) in enumerate(outputs_by_doc.items()):
        for method_index, method in enumerate(methods_per_doc):
            local_row, col = divmod(method_index, columns)
            row = doc_index * int(np.ceil(len(methods_per_doc) / columns)) + local_row
            tile = cv2.resize(outputs[method], (tile_w, tile_h - 28), interpolation=cv2.INTER_NEAREST)
            y0, x0 = row * tile_h, col * tile_w
            montage[y0 + 28:y0 + tile_h, x0:x0 + tile_w] = tile
            label = f"{doc_id}:{method}"[:28]
            cv2.putText(montage, label, (x0 + 3, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 0, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), montage):
        raise OSError(f"Montaj yazılamadı: {output_path}")


def _summarize(
    trial_values: dict[tuple[str, str, str], dict[str, list[float]]]
) -> list[dict[str, object]]:
    """Belge başına ve tüm belge×seed ölçümlerine mean/std satırları üretir."""
    rows: list[dict[str, object]] = []
    group_keys = sorted({(condition, method) for condition, method, _ in trial_values})
    for condition, method in group_keys:
        for doc_id in (*DOC_SPECS.keys(), "tum_belgeler"):
            keys = [
                key for key in trial_values
                if key[0] == condition and key[1] == method
                and (doc_id == "tum_belgeler" or key[2] == doc_id)
            ]
            metric_values: dict[str, list[float]] = {metric: [] for metric in ALL_METRICS}
            for key in keys:
                for metric, values in trial_values[key].items():
                    metric_values[metric].extend(values)
            row: dict[str, object] = {
                "Kosul": condition,
                "BelgeID": doc_id,
                "Yontem": method,
                "N": len(metric_values["F1"]),
                "Tohumlar": ";".join(map(str, SEEDS)),
            }
            for metric in ALL_METRICS:
                values = metric_values[metric]
                if values:
                    mean, std = _mean_std(values)
                    row[f"{metric}_ortalama"] = mean
                    row[f"{metric}_std"] = std
                else:
                    row[f"{metric}_ortalama"] = ""
                    row[f"{metric}_std"] = ""
            rows.append(row)
    return rows


def run() -> None:
    """Dört koşulu, üç sentetik belgeyi ve beş seed'i CSV/montaj olarak ölçer."""
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    montage_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {
        condition: {} for condition in CONDITIONS
    }

    for doc_id in DOC_SPECS:
        page_bgr, text_mask = _make_document(doc_id)
        clean = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        for condition in CONDITIONS:
            for seed in SEEDS:
                degraded = _condition_input(clean, condition, seed)
                enhanced, binary = _all_outputs(degraded)
                if seed == SEEDS[0]:
                    montage_outputs[condition][doc_id] = binary
                for method in METHOD_NAMES:
                    prediction = binary[method]
                    values = {
                        "F1": f1_score(text_mask, prediction),
                        "IoU": iou_score(text_mask, prediction),
                    }
                    if method in enhanced:
                        gray_output = enhanced[method]
                        values.update({
                            "PSNR": psnr(clean, gray_output),
                            "SSIM": ssim(clean, gray_output),
                            "Kontrast_std": float(gray_output.astype(np.float64).std()),
                            "Keskinlik_laplacian": sharpness_laplacian_var(gray_output),
                        })
                    group = (condition, method, doc_id)
                    for metric, value in values.items():
                        trial_values.setdefault(group, {}).setdefault(metric, []).append(float(value))

    for condition in CONDITIONS:
        _make_montage(
            montage_outputs[condition], condition,
            output_dir / f"enhancement_montage_{condition}.png",
        )

    rows = _summarize(trial_values)
    csv_path = output_dir / "enhancement_experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        loaded_rows = list(csv.DictReader(handle))
    summary_rows = [row for row in loaded_rows if row["BelgeID"] == "tum_belgeler"]
    figure = Figure(figsize=(22, 12), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for axis, metric in zip(axes.flat, ("PSNR", "SSIM", "F1", "IoU")):
        for condition in CONDITIONS:
            selected = [
                row for row in summary_rows
                if row["Kosul"] == condition and row[f"{metric}_ortalama"] != ""
            ]
            axis.plot(
                [row["Yontem"] for row in selected],
                [float(row[f"{metric}_ortalama"]) for row in selected],
                marker="o", label=condition,
            )
        axis.set_title(f"{metric}: ortalama (3 belge × 5 seed)")
        axis.tick_params(axis="x", labelrotation=70)
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "exp5_enhancement.png", dpi=150)

    print(f"CSV: {csv_path}")
    print("Kosul | Yontem | F1 ort±std | IoU ort±std | N")
    for row in summary_rows:
        print(
            f"{row['Kosul']} | {row['Yontem']} | "
            f"{float(row['F1_ortalama']):.4f}±{float(row['F1_std']):.4f} | "
            f"{float(row['IoU_ortalama']):.4f}±{float(row['IoU_std']):.4f} | {row['N']}"
        )


if __name__ == "__main__":
    run()
