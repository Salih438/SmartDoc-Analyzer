"""
src/preprocessing/enhancement.py
----------------------------------
Kontrast ve parlaklik iyilestirme algoritmalarini barindiron modul.

Dort farkli yontem karsilastirmali degerlendirme icin bir arada sunulur:
kontrast gerdirme, gama duzeltme, CLAHE ve global histogram esitleme.
Hangi yontemin belirli belge turunde daha iyi sonuc verdigi deneylerle
belirlenir; bu modul sonucu onceden sabitleme.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Yardimci
# ---------------------------------------------------------------------------

def _validate_gray(img: np.ndarray, func_name: str) -> None:
    """Girisin 2-D gri tonlamali dizi oldugunu dogrular."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{func_name}: np.ndarray bekleniyor, alinan {type(img)}")
    if img.ndim != 2:
        raise ValueError(
            f"{func_name}: 2-D gri goruntu bekleniyor, alinan sekil {img.shape}. "
            "Once to_grayscale() uygulayın."
        )


# ---------------------------------------------------------------------------
# 1. Kontrast Gerdirme (Percentile Stretching)
# ---------------------------------------------------------------------------

def contrast_stretch(
    gray: np.ndarray,
    low_pct: float = 2,
    high_pct: float = 98,
) -> np.ndarray:
    """Yuzdelik dilime dayali dogrusal kontrast gerdirme uygular.

    Problem
    -------
    Dusuk kontrastli belge goruntularinde piksel degerleri dar bir aralıga
    sikisir (or. 80-170). Bu fonksiyon, secilen alt/ust yuzdelikleri
    [0, 255] araligina yeniden esleyerek gorsel kontrasti artirir.
    Global histogram esitlemesinin aksine, asiri piksel degerlerini
    (yazici artefaktlari, gurultu noktalari) disarida birakmak icin
    yuzdelik kirpma secenegi sunar.

    Algoritma
    ---------
    1. low_pct ve high_pct yuzdeliklerini hesapla -> (p_low, p_high)
    2. Piksel degerlerini [p_low, p_high] araligina kliple
    3. Dogrusal yeniden olcekle: out = (img - p_low) / (p_high - p_low) * 255

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi (uint8 onerilir).
    low_pct : float, varsayilan 2
        Alt kirpma yuzdeligI. Artirilirsa daha fazla piksel 0'a
        kliplenir; belge kenarlarindaki koyu golgeleri bastirmak icin
        kullanislidir.
    high_pct : float, varsayilan 98
        Ust kirpma yuzdeligI. Dusurulurse daha fazla piksel 255'e
        kliplenir; parlak artefaktlari bastirmak icin kullanislidir.

    Dondurur
    --------
    np.ndarray
        uint8, sekil orijinalle ayni, piksel araligi [0, 255].

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse.
    - Tamamen tekduze goruntulerde (p_low == p_high) anlamli gerdirme
      yapilamaz; bu durumda girisin bir **kopyasi** dondurulur (siyah
      dizi degil). Bu davranis docstring'de acikca belirtilmistir.
    - uint8 disi dtype'larda sonuc dogru olabilir; ancak [0, 255] disi
      piksel degerleri beklenmedik kirpmaya yol acabilir.
    """
    _validate_gray(gray, "contrast_stretch")

    p_low = float(np.percentile(gray, low_pct))
    p_high = float(np.percentile(gray, high_pct))

    if p_high <= p_low:
        # Tamamen tekduze goruntu: anlamli gerdirme yapilamaz; kopya dondur
        return gray.copy()

    stretched = (gray.astype(np.float64) - p_low) / (p_high - p_low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Gama Duzeltme (Gamma Correction)
# ---------------------------------------------------------------------------

def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Gama duzeltme ile goruntu parlakligi dogrusal olmayan bicimde ayarlar.

    Problem
    -------
    Bazi belge tarayicilari veya kamera kosullari, goruntuyu gerektigi kadar
    koyu veya acik uretir. Gama duzeltme, [0, 255] araligindaki piksellere
    asagidaki formul ile donusum uygulayarak bu dengesizligi telafi eder.

    Formul
    ------
    out = 255 * (in / 255) ^ (1 / gamma)

    Yon
    ---
    - gamma > 1 : us < 1  ->  goruntu AYDINLATIR (koyu bolgeleri acar)
    - gamma < 1 : us > 1  ->  goruntu KOYULASTIRIR (acik bolgeleri karatir)
    - gamma = 1 : us = 1  ->  degisiklik yok

    Algoritma
    ---------
    1. 256 elemanli LUT (Lookup Table) olustur:
       LUT[i] = clip(((i / 255) ** (1/gamma)) * 255, 0, 255)
    2. LUT uzerinden goruntu donustur -> O(1) piksel basina, hizli.

    Neden LUT?
    ----------
    Her piksele ayri ayri us alma yerine 256 deger icin tek seferlik
    hesaplama yapilir; boylece buyuk goruntulerde onemli hiz kazanci saglanir.

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi (uint8 beklenir).
    gamma : float
        Gama katsayisi. Uygulanan us (1/gamma) oldugu icin:
        - gamma > 1 : Goruntu aydınlatiR (us < 1).
        - 0 < gamma < 1 : Goruntu koyulastirir (us > 1).
        - gamma = 1 : Degisiklik yok (us = 1).

    Dondurur
    --------
    np.ndarray
        uint8, sekil orijinalle ayni.

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse veya gamma <= 0 ise.
    - Cok kucuk gamma degerleri (< 0.01) icin lut hesabinda buyuk usler
      olusur; pratikle 0.1-5.0 araligi guvenlidir.
    """
    _validate_gray(gray, "gamma_correction")

    if gamma <= 0:
        raise ValueError(f"gamma pozitif olmalidir, alinan: {gamma}")

    lut = np.array(
        [np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, lut)


# ---------------------------------------------------------------------------
# 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------------------------

def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE algoritmasiyla yerel kontrast iyilestirme uygular.

    Problem
    -------
    Global histogram esitleme, gOruntunun tamamindaki dagilima gore
    calistigindan yerel koyu/acik bolgelerde asiri iyilestirme veya
    gurultu amplifikasyonu uretebilir. CLAHE, goruntu kucuk karolara
    boler, her karoda sinirlandirilmis histogram esitleme uygular ve
    karolar arasi interpolasyon ile dogal gecisler saglar.

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi (uint8 beklenir).
    clip_limit : float, varsayilan 2.0
        Histogram karolari icin kontrast siniri. Artirilirsa daha guclu
        kontrast iyilestirme saglanir, ancak gurultu de artar.
        Azaltilirsa daha muhafazakar iyilestirme yapilir.
    tile_grid_size : tuple[int, int], varsayilan (8, 8)
        Goruntunun bolunecegi yatay ve dikey karo sayisi.
        Buyuk degerler (or. 16, 16) daha yerel uyarlamayi; kucuk degerler
        (or. 2, 2) daha kuresel davranisi simule eder.

    Dondurur
    --------
    np.ndarray
        uint8, giris goruntusiyle ayni sekil ve dtype.

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse.
    - clip_limit cok buyuk secilirse (> 40) gurultu asiri yukseltilir;
      belge metin kalitesi dusebilir.
    - tile_grid_size boyutlari goruntu boyutundan buyukse OpenCV hata verir.
    """
    _validate_gray(gray, "apply_clahe")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# 4. Global Histogram Esitleme
# ---------------------------------------------------------------------------

def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Kuresel histogram esitleme uygular (karsilastirma referansi).

    Problem
    -------
    Dar yogunluk dagilimina sahip dusuk kontrastli goruntulerde, pikselleri
    [0, 255] araligina yayarak kontrast arttirir. CLAHE ile karsilastirma
    amaciyla referans algoritma olarak sunulur; asiri amplifikasyon ve
    gurultu artisi riski CLAHE'ye kiyasla daha yuksektir.

    Algoritma
    ---------
    1. CDF (kumulatif dagilim fonksiyonu) hesapla
    2. CDF'yi [0, 255] araligina dogrusal olarak esle
    3. Her pikseli yeni yogunluk degeriyle degistir

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi (uint8 beklenir).

    Dondurur
    --------
    np.ndarray
        uint8, giris goruntusiyle ayni sekil.

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse.
    - Tekduze goruntulerde (tum pikseller ayni deger) cv2.equalizeHist
      degeri DEGISTIRMEZ; giris degeri korunur (ornegin 128 -> 128 kalir).
      Bu OpenCV'nin beklenen davranisidir; 255'e donusmez.
    - Arka plani buyuk, metin alani kucuk olan belgelerde metin bolgeleri
      asiri karatilebilir; bu durumda CLAHE tercih edilmelidir.
    """
    _validate_gray(gray, "histogram_equalization")
    return cv2.equalizeHist(gray)