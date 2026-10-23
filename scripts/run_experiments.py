"""
scripts/run_experiments.py
--------------------------
SmartDoc Analyzer Asama 2 Deney Betigi.

Gercekci cozunurlukte (1650x2200) sentetik belge ve referans metin maskesi
(ground_truth_text_mask) uzerinde 5 temel deneyi calistirir:
  1. Filtreler: Gaussian, Median ve Bilateral (filtresiz baseline dahil)
  2. Unsharp Masking: Onceden bulaniklastirilmis belgede amount varyasyonu (0.5-5.0)
  3. Esikleme: Normal, Dengesiz (0.4, 0.7, 0.9), Gurultulu+Dengesiz ve filtreli girdi
     Precision, Recall, F1, IoU ve Foreground Ratio metrikleri
  4. Morfoloji: Gurultulu ikili goruntude opening/closing (ksize=3 ve 5) F1/IoU

Sonuclar results/metrics.csv ve results/*.png olarak kaydedilir.
Yontem ustunlugune dair yorum yapilmaz; yalnizca tarafsiz sayisal veriler sunulur.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Proje kok dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.image_info import save_image
from src.preprocessing.filtering import gaussian_blur, median_blur, bilateral_filter
from src.preprocessing.sharpening import unsharp_mask
from src.segmentation.threshold import global_threshold, otsu_threshold, adaptive_threshold
from src.segmentation.morphology import opening, closing
from src.evaluation.metrics import (
    psnr,
    ssim,
    sharpness_laplacian_var,
    foreground_ratio,
    precision,
    recall,
    f1_score,
    iou_score,
    timed_median,
)
from src.evaluation.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    simulate_uneven_illumination,
)


def create_synthetic_document(width: int = 1650, height: int = 2200) -> tuple[np.ndarray, np.ndarray]:
    """Gerçekçi belge ve bağımsız çizilmiş ikili metin GT maskesi üretir.

    Belgeye metinler LINE_AA ile çizilir. GT, aynı geometrik çizim tarifleri
    kullanılarak ayrı bir 4x çözünürlüklü maskede LINE_8 ile çizilir; piksel
    alanı en az %50 kaplıysa çıktı maskesi 255, değilse 0 olur. GT, belge
    görüntüsünü eşikleyerek türetilmez.

    Dondurur
    --------
    tuple[np.ndarray, np.ndarray]
        (doc, ground_truth_text_mask)
        doc: beyaz zemin (255) uzerine siyah yazi (0).
        ground_truth_text_mask: yazi pikselleri 255, kagit zemin 0.
    """
    supersample = 4
    doc_hi = np.full((height * supersample, width * supersample), 255, dtype=np.uint8)
    mask_hi = np.zeros((height * supersample, width * supersample), dtype=np.uint8)

    def point4(point: tuple[int, int]) -> tuple[int, int]:
        return point[0] * supersample, point[1] * supersample

    def draw_text(txt: str, pos: tuple[int, int], scale: float = 1.6, thickness: int = 3) -> None:
        cv2.putText(doc_hi, txt, point4(pos), cv2.FONT_HERSHEY_SIMPLEX, scale * supersample, 0, thickness * supersample, cv2.LINE_AA)
        cv2.putText(mask_hi, txt, point4(pos), cv2.FONT_HERSHEY_SIMPLEX, scale * supersample, 255, thickness * supersample, cv2.LINE_8)

    def draw_line(p1: tuple[int, int], p2: tuple[int, int], thickness: int = 3) -> None:
        cv2.line(doc_hi, point4(p1), point4(p2), 0, thickness * supersample, cv2.LINE_AA)
        cv2.line(mask_hi, point4(p1), point4(p2), 255, thickness * supersample, cv2.LINE_8)

    def draw_rect(p1: tuple[int, int], p2: tuple[int, int], thickness: int = 3) -> None:
        cv2.rectangle(doc_hi, point4(p1), point4(p2), 0, thickness * supersample, cv2.LINE_AA)
        cv2.rectangle(mask_hi, point4(p1), point4(p2), 255, thickness * supersample, cv2.LINE_8)

    # 1. Baslik ve ust bilgi
    draw_text("SMARTDOC ANALYZER - ARASTIRMA RAPORU", (110, 140), scale=2.0, thickness=4)
    draw_line((110, 175), (width - 110, 175), thickness=4)

    draw_text("Belge No: 2026-EXP-02 / YUKSEK COZUNURLUK", (110, 250), scale=1.4, thickness=3)
    draw_text("Tarih: 17.10.2026", (110, 310), scale=1.4, thickness=3)
    draw_text("Konu: Klasik Bilgisayarli Goru ve Segmentasyon Deneyleri", (110, 370), scale=1.4, thickness=3)

    # 2. Bolumler ve paragraflar
    paragraphs = [
        "1. ALTYAPI VE PROBLEM TANIMI",
        "Bu rapor, taranmis ve mobil aygitlarla fotograflanmis metin belgelerinin analizinde",
        "klasik goruntu isleme algoritmalarinin nicel performansini ortaya koymaktadir.",
        "Goruntuler cozunurluk kayiplarina, duzensiz aydinlatmaya, sensor gurultusune ve",
        "kagit bozulmalarina karsi test edilmektedir.",
        "",
        "2. YONTEMLER VE PARAMETRELER",
        "Calisma kapsaminda Gaussian Blur, Median Blur ve Bilateral Filtre karsilastirilmis;",
        "Unsharp Masking ile harf kenarlari netlestirilmis; Global, Otsu ve Uyarlanabilir",
        "esikleme ile metin binarizasyonu saglanmis; morfolojik filtrelerle leke ve delikler",
        "temizlenmis; Canny ve Auto-Canny ile yapi kenarlari cikarilmistir.",
        "",
        "3. PERFORMANS VE KONTROL TABLOSU",
    ]

    y = 480
    for line in paragraphs:
        if line:
            is_header = line.startswith(("1.", "2.", "3."))
            sc = 1.6 if is_header else 1.35
            th = 4 if is_header else 3
            draw_text(line, (110, y), scale=sc, thickness=th)
        y += 65

    # 3. Tablo
    t_top = y + 20
    t_bot = t_top + 360
    draw_rect((110, t_top), (width - 110, t_bot), thickness=4)
    draw_line((110, t_top + 90), (width - 110, t_top + 90), thickness=3)
    draw_line((110, t_top + 180), (width - 110, t_top + 180), thickness=2)
    draw_line((110, t_top + 270), (width - 110, t_top + 270), thickness=2)

    draw_line((480, t_top), (480, t_bot), thickness=3)
    draw_line((1020, t_top), (1020, t_bot), thickness=3)

    draw_text("Parametre", (140, t_top + 62), scale=1.4, thickness=3)
    draw_text("Hedef Islem", (520, t_top + 62), scale=1.4, thickness=3)
    draw_text("Degerlendirme Metrigi", (1060, t_top + 62), scale=1.4, thickness=3)

    draw_text("ksize = 5", (140, t_top + 150), scale=1.3, thickness=3)
    draw_text("Gurultu Azaltma", (520, t_top + 150), scale=1.3, thickness=3)
    draw_text("PSNR (dB) / SSIM", (1060, t_top + 150), scale=1.3, thickness=3)

    draw_text("amount = 1.0", (140, t_top + 240), scale=1.3, thickness=3)
    draw_text("Bulaniklik Giderme", (520, t_top + 240), scale=1.3, thickness=3)
    draw_text("Laplacian Varyansi", (1060, t_top + 240), scale=1.3, thickness=3)

    draw_text("block = 31", (140, t_top + 330), scale=1.3, thickness=3)
    draw_text("Metin Binarizasyonu", (520, t_top + 330), scale=1.3, thickness=3)
    draw_text("F1-Score / IoU", (1060, t_top + 330), scale=1.3, thickness=3)

    # 4. Alt bilgi ve onay alani
    draw_line((110, height - 180), (width - 110, height - 180), thickness=3)
    draw_text("SmartDoc Analyzer v2.0 - Sentetik Belge Analiz Sistemi", (110, height - 110), scale=1.2, thickness=2)
    draw_text("Laboratuvar Onayi: [KLASIK-CV-LAB]", (width - 650, height - 110), scale=1.2, thickness=2)

    # En az %50 yüksek çözünürlüklü kapsama, GT'de 255 kabul edilir.
    doc = cv2.resize(doc_hi, (width, height), interpolation=cv2.INTER_AREA)
    coverage = cv2.resize(mask_hi, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.where(coverage >= 127.5, 255, 0).astype(np.uint8)
    assert set(np.unique(mask)).issubset({0, 255})
    return doc, mask


def make_visual_grid(images: list[np.ndarray], labels: list[str], max_cols: int = 3, target_w: int = 550) -> np.ndarray:
    """Yuksek cozunurluklu gorselleri makul boyutta birlestirir."""
    cells = []
    aspect = images[0].shape[0] / images[0].shape[1]
    cell_h = int(target_w * aspect)

    for img, lbl in zip(images, labels):
        # 1 kanalli gri ise 3 kanala cevir (etiket cubugu icin)
        res = cv2.resize(img, (target_w, cell_h), interpolation=cv2.INTER_AREA)
        if res.ndim == 2:
            res_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        else:
            res_bgr = res.copy()

        header = np.full((40, target_w, 3), 240, dtype=np.uint8)
        cv2.putText(header, lbl, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cells.append(np.vstack([header, res_bgr]))

    # Izgara olustur
    rows = []
    tot_h, tot_w = cells[0].shape[:2]
    for i in range(0, len(cells), max_cols):
        chunk = cells[i : i + max_cols]
        while len(chunk) < max_cols:
            chunk.append(np.full((tot_h, tot_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(chunk))

    return np.vstack(rows)


def run() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Önceki çalıştırmanın GT karşılaştırmasını yeniden çalıştırmalar arasında koru.
    comparison_path = results_dir / "gt_mask_comparison.csv"
    old_mask_difference: str | None = None
    if comparison_path.exists():
        with comparison_path.open(newline="", encoding="utf-8") as handle:
            prior = next(csv.DictReader(handle), None)
            if prior:
                old_mask_difference = prior.get("Eski_doc127_mask_fark_px") or None
    elif (results_dir / "metrics.csv").exists():
        with (results_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            prior = next(csv.DictReader(handle), None)
            if prior:
                old_mask_difference = prior.get("GT_mask_doc127_fark_px") or None

    print("=" * 90)
    print(" SmartDoc Analyzer - Asama 2 Deneyleri (Guncel Tasarim) Baslatiliyor")
    print("=" * 90)

    # 1. Sentetik Belge Uretimi
    doc, gt_mask = create_synthetic_document()
    doc_threshold_foreground = doc <= 127
    gt_mask_foreground = gt_mask > 0
    gt_mask_difference = int(np.count_nonzero(doc_threshold_foreground != gt_mask_foreground))
    independently_counted_difference = int(np.count_nonzero(np.logical_xor(doc <= 127, gt_mask > 0)))
    assert gt_mask_difference == independently_counted_difference
    doc_foreground_count = int(np.count_nonzero(doc_threshold_foreground))
    mask_foreground_count = int(np.count_nonzero(gt_mask_foreground))
    shared_foreground_count = int(np.count_nonzero(doc_threshold_foreground & gt_mask_foreground))
    gt_mask_difference_ratio = gt_mask_difference / gt_mask.size
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("Maske_tanimi", "Eski_doc127_mask_fark_px", "Yeni_doc127_mask_fark_px", "Yeni_fark_orani", "doc127_onplan_px", "GT_onplan_px", "Ortak_onplan_px", "doc_ozel_px", "GT_ozel_px", "Karsilastirma_kaynagi")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Maske_tanimi": "4x LINE_8 çizim; kapsama >= %50 ise 255",
            "Eski_doc127_mask_fark_px": old_mask_difference or "önceki CSV bulunamadı",
            "Yeni_doc127_mask_fark_px": gt_mask_difference,
            "Yeni_fark_orani": f"{gt_mask_difference_ratio:.8%}",
            "doc127_onplan_px": doc_foreground_count,
            "GT_onplan_px": mask_foreground_count,
            "Ortak_onplan_px": shared_foreground_count,
            "doc_ozel_px": doc_foreground_count - shared_foreground_count,
            "GT_ozel_px": mask_foreground_count - shared_foreground_count,
            "Karsilastirma_kaynagi": "doc <= 127 ile bağımsız GT mask > 0",
        })
    save_image(results_dir / "00_synthetic_doc.png", doc)
    save_image(results_dir / "00_ground_truth_mask.png", gt_mask)
    print(f"[1/5] Sentetik belge (1650x2200) uretildi. Yazi pikselleri: {int(np.count_nonzero(gt_mask == 255))}")
    print(
        "GT maskesi doc'tan bağımsız çizildi; doc<=127 ile fark: "
        f"{gt_mask_difference} piksel ({gt_mask_difference_ratio:.8%})"
    )
    print(
        f"doc<=127 önplan={doc_foreground_count}, GT önplan={mask_foreground_count}, "
        f"ortak={shared_foreground_count}, doc-özel={doc_foreground_count - shared_foreground_count}, "
        f"GT-özel={mask_foreground_count - shared_foreground_count}"
    )

    records: list[dict[str, Any]] = []

    def add_row(
        deney: str,
        yontem: str,
        param: str = "",
        psnr_val: float | None = None,
        ssim_val: float | None = None,
        prec_val: float | None = None,
        rec_val: float | None = None,
        f1_val: float | None = None,
        iou_val: float | None = None,
        sure_val: float | None = None,
    ) -> None:
        records.append({
            "Deney": deney,
            "Yontem": yontem,
            "Parametre": param,
            "PSNR_dB": f"{psnr_val:.3f}" if psnr_val is not None else "",
            "SSIM": f"{ssim_val:.4f}" if ssim_val is not None else "",
            "Precision": f"{prec_val:.4f}" if prec_val is not None else "",
            "Recall": f"{rec_val:.4f}" if rec_val is not None else "",
            "F1": f"{f1_val:.4f}" if f1_val is not None else "",
            "IoU": f"{iou_val:.4f}" if iou_val is not None else "",
            "Sure_ms_medyan": f"{sure_val:.2f}" if sure_val is not None else "",
            "GT_mask_doc127_fark_px": gt_mask_difference,
            "GT_mask_doc127_fark_oran": f"{gt_mask_difference_ratio:.8f}",
        })

    # -----------------------------------------------------------------------
    # DENEY 1: Gaussian vs Median vs Bilateral (Baseline Dahil)
    # -----------------------------------------------------------------------
    print("[2/5] Deney 1: Gurultu Filtreleme (Baseline dahil)...")

    # 1A: Gaussian Gurultusu
    noisy_gauss = add_gaussian_noise(doc, sigma=25.0, seed=42)
    psnr_gn = psnr(doc, noisy_gauss)
    ssim_gn = ssim(doc, noisy_gauss)
    add_row("1A_Filtre_Gaussian", "Gurultulu (Filtresiz)", "sigma=25 (Baseline)", psnr_val=psnr_gn, ssim_val=ssim_gn)

    g_out, t_g = timed_median(gaussian_blur, noisy_gauss, ksize=5, repeats=10, warmup=2)
    add_row("1A_Filtre_Gaussian", "Gaussian Blur", "ksize=5", psnr(doc, g_out), ssim(doc, g_out), sure_val=t_g)

    m_out, t_m = timed_median(median_blur, noisy_gauss, ksize=5, repeats=10, warmup=2)
    add_row("1A_Filtre_Gaussian", "Median Blur", "ksize=5", psnr(doc, m_out), ssim(doc, m_out), sure_val=t_m)

    b_out, t_b = timed_median(bilateral_filter, noisy_gauss, d=9, sigma_color=75, sigma_space=75, repeats=5, warmup=1)
    add_row("1A_Filtre_Gaussian", "Bilateral Filter", "d=9, sc=75, ss=75", psnr(doc, b_out), ssim(doc, b_out), sure_val=t_b)

    grid_1a = make_visual_grid(
        [doc, noisy_gauss, g_out, m_out, b_out],
        ["Temiz Belge", "Gaussian Gurultulu", "Gaussian Blur", "Median Blur", "Bilateral Filter"],
        max_cols=3,
    )
    save_image(results_dir / "exp1a_gaussian_noise.png", grid_1a)

    # 1B: Tuz-Biber Gurultusu
    noisy_sp = add_salt_pepper_noise(doc, amount=0.06, seed=42)
    psnr_sp = psnr(doc, noisy_sp)
    ssim_sp = ssim(doc, noisy_sp)
    add_row("1B_Filtre_SaltPepper", "Gurultulu (Filtresiz)", "amount=0.06 (Baseline)", psnr_val=psnr_sp, ssim_val=ssim_sp)

    g_sp, t_g_sp = timed_median(gaussian_blur, noisy_sp, ksize=5, repeats=10, warmup=2)
    add_row("1B_Filtre_SaltPepper", "Gaussian Blur", "ksize=5", psnr(doc, g_sp), ssim(doc, g_sp), sure_val=t_g_sp)

    m_sp, t_m_sp = timed_median(median_blur, noisy_sp, ksize=5, repeats=10, warmup=2)
    add_row("1B_Filtre_SaltPepper", "Median Blur", "ksize=5", psnr(doc, m_sp), ssim(doc, m_sp), sure_val=t_m_sp)

    b_sp, t_b_sp = timed_median(bilateral_filter, noisy_sp, d=9, sigma_color=75, sigma_space=75, repeats=5, warmup=1)
    add_row("1B_Filtre_SaltPepper", "Bilateral Filter", "d=9, sc=75, ss=75", psnr(doc, b_sp), ssim(doc, b_sp), sure_val=t_b_sp)

    grid_1b = make_visual_grid(
        [doc, noisy_sp, g_sp, m_sp, b_sp],
        ["Temiz Belge", "Tuz-Biber Gurultulu", "Gaussian Blur", "Median Blur", "Bilateral Filter"],
        max_cols=3,
    )
    save_image(results_dir / "exp1b_salt_pepper_noise.png", grid_1b)

    # -----------------------------------------------------------------------
    # DENEY 2: Unsharp Masking (Once Bulaniklastirma -> Sonra Keskinlestirme)
    # -----------------------------------------------------------------------
    print("[3/5] Deney 2: Unsharp Masking (Bulanik girdi baseline ile)...")
    unsharp_vis = [doc]
    unsharp_labels = [f"Orijinal (Var={sharpness_laplacian_var(doc):.1f})"]

    for blur_sigma in (1.5, 2.5):
        blurred = cv2.GaussianBlur(doc, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        var_bl = sharpness_laplacian_var(blurred)
        add_row(
            f"2_Unsharp_sigma{blur_sigma}",
            "Bulanik Girdi",
            f"Gaussian sigma={blur_sigma} (Baseline), Var={var_bl:.1f}",
            psnr_val=psnr(doc, blurred),
            ssim_val=ssim(doc, blurred),
        )
        unsharp_vis.append(blurred)
        unsharp_labels.append(f"Bulanik s={blur_sigma} (Var={var_bl:.1f})")

        for amt in (0.5, 1.0, 2.0, 3.0, 5.0):
            res_amt, t_amt = timed_median(unsharp_mask, blurred, ksize=5, sigma=1.0, amount=amt, repeats=10, warmup=2)
            var_amt = sharpness_laplacian_var(res_amt)
            add_row(
                f"2_Unsharp_sigma{blur_sigma}",
                f"Unsharp (amount={amt})",
                f"amount={amt}, Var={var_amt:.1f}",
                psnr_val=psnr(doc, res_amt),
                ssim_val=ssim(doc, res_amt),
                sure_val=t_amt,
            )
            unsharp_vis.append(res_amt)
            unsharp_labels.append(f"s={blur_sigma} amt={amt} (Var={var_amt:.1f})")

    grid_2 = make_visual_grid(unsharp_vis, unsharp_labels, max_cols=3)
    save_image(results_dir / "exp2_unsharp_amounts.png", grid_2)

    # -----------------------------------------------------------------------
    # DENEY 3: Esikleme Olcumu (Precision, Recall, F1, IoU - invert=True)
    # -----------------------------------------------------------------------
    print("[4/5] Deney 3: Esikleme Olcumu (F1, IoU, Farkli Isik Kosullari)...")
    chosen_direction = np.random.default_rng(42).choice(["diag", "horizontal", "vertical"])
    threshold_scenarios = [
        ("3A_Normal", doc, "Normal Aydinlatma"),
        ("3B_Uneven_0.4", simulate_uneven_illumination(doc, strength=0.4, seed=42), f"Dengesiz Isik (str=0.4, yön={chosen_direction})"),
        ("3C_Uneven_0.7", simulate_uneven_illumination(doc, strength=0.7, seed=42), f"Dengesiz Isik (str=0.7, yön={chosen_direction})"),
        ("3D_Uneven_0.9", simulate_uneven_illumination(doc, strength=0.9, seed=42), f"Dengesiz Isik (str=0.9, yön={chosen_direction})"),
        ("3E_Noisy_Uneven", add_gaussian_noise(simulate_uneven_illumination(doc, strength=0.7, seed=42), sigma=20.0, seed=42), f"Gurultu+Dengesiz (str=0.7, s=20, yön={chosen_direction})"),
        ("3F_Filtered_Noisy_Uneven", bilateral_filter(median_blur(add_gaussian_noise(simulate_uneven_illumination(doc, strength=0.7, seed=42), sigma=20.0, seed=42), ksize=5), d=9, sigma_color=75, sigma_space=75), "Median(5)+Bilateral, eşiklemeden önce"),
    ]

    thresh_vis_all = []
    thresh_labels_all = []

    for sc_name, sc_img, sc_desc in threshold_scenarios:
        # 1. Global (127)
        bin_glob, t_glob = timed_median(global_threshold, sc_img, thresh=127, invert=True, repeats=15, warmup=2)
        add_row(
            sc_name,
            "Global (127)",
            f"{sc_desc}, fg={foreground_ratio(bin_glob):.4f}",
            prec_val=precision(gt_mask, bin_glob),
            rec_val=recall(gt_mask, bin_glob),
            f1_val=f1_score(gt_mask, bin_glob),
            iou_val=iou_score(gt_mask, bin_glob),
            sure_val=t_glob,
        )

        # 2. Otsu
        (bin_otsu, val_otsu), t_otsu = timed_median(otsu_threshold, sc_img, invert=True, repeats=15, warmup=2)
        add_row(
            sc_name,
            "Otsu",
            f"esik={val_otsu:.1f}, fg={foreground_ratio(bin_otsu):.4f}",
            prec_val=precision(gt_mask, bin_otsu),
            rec_val=recall(gt_mask, bin_otsu),
            f1_val=f1_score(gt_mask, bin_otsu),
            iou_val=iou_score(gt_mask, bin_otsu),
            sure_val=t_otsu,
        )

        # 3. Adaptive Mean (block=31, c=10)
        bin_adm, t_adm = timed_median(adaptive_threshold, sc_img, method="mean", block_size=31, c=10, invert=True, repeats=10, warmup=2)
        add_row(
            sc_name,
            "Adaptive Mean",
            f"block=31, c=10, fg={foreground_ratio(bin_adm):.4f}",
            prec_val=precision(gt_mask, bin_adm),
            rec_val=recall(gt_mask, bin_adm),
            f1_val=f1_score(gt_mask, bin_adm),
            iou_val=iou_score(gt_mask, bin_adm),
            sure_val=t_adm,
        )

        # 4. Adaptive Gaussian (block=31, c=10)
        bin_adg, t_adg = timed_median(adaptive_threshold, sc_img, method="gaussian", block_size=31, c=10, invert=True, repeats=10, warmup=2)
        add_row(
            sc_name,
            "Adaptive Gaussian",
            f"block=31, c=10, fg={foreground_ratio(bin_adg):.4f}",
            prec_val=precision(gt_mask, bin_adg),
            rec_val=recall(gt_mask, bin_adg),
            f1_val=f1_score(gt_mask, bin_adg),
            iou_val=iou_score(gt_mask, bin_adg),
            sure_val=t_adg,
        )

        thresh_vis_all.extend([sc_img, bin_glob, bin_otsu, bin_adm, bin_adg])
        thresh_labels_all.extend([sc_desc, f"Global ({sc_name})", f"Otsu {val_otsu:.0f}", f"AdMean ({sc_name})", f"AdGauss ({sc_name})"])

    grid_3 = make_visual_grid(thresh_vis_all[:10], thresh_labels_all[:10], max_cols=5)
    save_image(results_dir / "exp3a_threshold_normal_uneven.png", grid_3)
    grid_3_extra = make_visual_grid(thresh_vis_all[10:], thresh_labels_all[10:], max_cols=5)
    save_image(results_dir / "exp3b_threshold_severe_noise.png", grid_3_extra)

    # -----------------------------------------------------------------------
    # DENEY 4: Morfoloji (ksize=3 ve 5 ile F1 ve IoU)
    # -----------------------------------------------------------------------
    print("[5/5] Deney 4: Morfoloji deneyleri...")
    # Referans maske uzerine tuz-biber gurultusu ekle
    noisy_bin = add_salt_pepper_noise(gt_mask, amount=0.04, seed=42)
    add_row(
        "4_Morfoloji",
        "Gurultulu Ikili (Baseline)",
        f"amount=0.04, fg={foreground_ratio(noisy_bin):.4f}",
        prec_val=precision(gt_mask, noisy_bin),
        rec_val=recall(gt_mask, noisy_bin),
        f1_val=f1_score(gt_mask, noisy_bin),
        iou_val=iou_score(gt_mask, noisy_bin),
    )

    morph_vis = [gt_mask, noisy_bin]
    morph_labels = ["Temiz Maske (GT)", "Gurultulu Ikili (sp=0.04)"]

    for k in (3, 5):
        op_out, t_op = timed_median(opening, noisy_bin, ksize=k, repeats=15, warmup=2)
        add_row(
            "4_Morfoloji",
            f"Opening (k={k})",
            f"ksize={k}, fg={foreground_ratio(op_out):.4f}",
            prec_val=precision(gt_mask, op_out),
            rec_val=recall(gt_mask, op_out),
            f1_val=f1_score(gt_mask, op_out),
            iou_val=iou_score(gt_mask, op_out),
            sure_val=t_op,
        )

        cl_out, t_cl = timed_median(closing, noisy_bin, ksize=k, repeats=15, warmup=2)
        add_row(
            "4_Morfoloji",
            f"Closing (k={k})",
            f"ksize={k}, fg={foreground_ratio(cl_out):.4f}",
            prec_val=precision(gt_mask, cl_out),
            rec_val=recall(gt_mask, cl_out),
            f1_val=f1_score(gt_mask, cl_out),
            iou_val=iou_score(gt_mask, cl_out),
            sure_val=t_cl,
        )

        op_cl_fn = lambda b: closing(opening(b, ksize=k), ksize=k)
        op_cl_out, t_op_cl = timed_median(op_cl_fn, noisy_bin, repeats=10, warmup=2)
        add_row(
            "4_Morfoloji",
            f"Opening+Closing (k={k})",
            f"ksize={k}, fg={foreground_ratio(op_cl_out):.4f}",
            prec_val=precision(gt_mask, op_cl_out),
            rec_val=recall(gt_mask, op_cl_out),
            f1_val=f1_score(gt_mask, op_cl_out),
            iou_val=iou_score(gt_mask, op_cl_out),
            sure_val=t_op_cl,
        )

        morph_vis.extend([op_out, cl_out, op_cl_out])
        morph_labels.extend([f"Opening (k={k})", f"Closing (k={k})", f"Open+Close (k={k})"])

    grid_4 = make_visual_grid(morph_vis, morph_labels, max_cols=4)
    save_image(results_dir / "exp4_morphology.png", grid_4)

    # -----------------------------------------------------------------------
    # CSV Kayit
    # -----------------------------------------------------------------------
    csv_path = results_dir / "metrics.csv"
    fieldnames = ["Deney", "Yontem", "Parametre", "PSNR_dB", "SSIM", "Precision", "Recall", "F1", "IoU", "Sure_ms_medyan", "GT_mask_doc127_fark_px", "GT_mask_doc127_fark_oran"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Tum deney sonuclari kaydedildi -> {csv_path}")
    print()

    # -----------------------------------------------------------------------
    # Terminal Tablo Ciktisi (Yorum yapilmadan yalnizca sayisal tablo)
    # -----------------------------------------------------------------------
    print("=" * 130)
    print(f"{'Deney':<18} | {'Yontem':<25} | {'PSNR':<7} | {'SSIM':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'IoU':<7} | {'Sure(ms)':<8} | {'Parametre / Not'}")
    print("-" * 130)
    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = csv.DictReader(f)
        for r in csv_rows:
            print(f"{r['Deney']:<18} | {r['Yontem']:<25} | {r['PSNR_dB']:<7} | {r['SSIM']:<7} | {r['Precision']:<7} | {r['Recall']:<7} | {r['F1']:<7} | {r['IoU']:<7} | {r['Sure_ms_medyan']:<8} | {r['Parametre']}")
    print("=" * 130)


if __name__ == "__main__":
    run()
