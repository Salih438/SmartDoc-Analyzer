"""
src/analysis/image_info.py
--------------------------
Goruntu yukleme, kaydetme ve temel istatistiksel bilgi cikarma modulu.

Windows'ta Turkce/Unicode karakterli dizin yollarinda cv2.imread/imwrite
calismayadigindan np.fromfile + cv2.imdecode / cv2.imencode + tofile
kullanilir.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Yukleme
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode PNG/JPG/JPEG bytes as a BGR uint8 image.

    Problem
    -------
    Streamlit uploads are in-memory bytes; decoding them must not depend on a
    temporary filename or a platform-specific path encoding.

    Algorithm
    ---------
    Validate the PNG/JPEG signature, view the bytes as uint8 with
    ``np.frombuffer``, then decode with OpenCV's ``imdecode``.

    Parameters
    ----------
    data : bytes
        Complete contents of a PNG, JPG, or JPEG image.

    Returns
    -------
    np.ndarray
        Decoded three-channel BGR image.

    Raises
    ------
    ValueError
        For empty data, unsupported formats, or corrupt/incomplete images.
    """
    if not data:
        raise ValueError("Görüntü verisi boş; PNG veya JPG/JPEG dosyası seçin.")
    if not (data.startswith(_PNG_SIGNATURE) or data.startswith(b"\xff\xd8\xff")):
        raise ValueError("Desteklenmeyen veya bozuk dosya; yalnızca PNG, JPG ve JPEG desteklenir.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Görüntü çözümlenemedi; dosya bozuk veya eksik olabilir.")
    return image

def load_image(path: str | Path) -> np.ndarray:
    """Diskten bir goruntu dosyasi yukler ve BGR NumPy dizisi olarak dondurur.

    Problem
    -------
    Windows'ta Turkce / Unicode karakterli dizin yollarinda cv2.imread
    None dondurebilir. Bu fonksiyon np.fromfile + cv2.imdecode kullanarak
    platformdan bagimsiz, karakter kodlamasi sorunsuz yukleme saglar.

    Parametreler
    ------------
    path : str | Path
        Goruntu dosyasinin yolu. PNG, JPG ve JPEG uzantilari desteklenir.

    Dondurur
    --------
    np.ndarray
        BGR renk uzayinda uint8 goruntu dizisi.  Sekil: (H, W, 3) veya
        alpha kanalli dosyalar icin (H, W, 4).

    Basarisiz Olabilir
    ------------------
    - FileNotFoundError : Dosya belirtilen yolda yoksa.
    - ValueError        : Dosya uzantisi desteklenmiyorsa veya OpenCV
                          goruntuyu decode edemiyorsa (bozuk dosya vb.).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: '{path}'")

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Desteklenmeyen dosya uzantısı: '{path.suffix}'. "
            f"Desteklenen uzantılar: {_SUPPORTED_EXTENSIONS}"
        )

    # Unicode-safe yukleme: dosyayi ham bayt olarak oku, ardindan decode et
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(
            f"OpenCV goruntuyu decode edemedi: '{path}'. "
            "Dosya bozuk veya desteklenmeyen bir ic format iceriyor olabilir."
        )

    return img


# ---------------------------------------------------------------------------
# Kaydetme
# ---------------------------------------------------------------------------

def save_image(path: str | Path, img: np.ndarray) -> None:
    """Bir goruntu dizisini diske kaydeder.

    Problem
    -------
    Windows'ta Unicode karakterli dizin yollarinda cv2.imwrite basarisiz
    olabilir. Bu fonksiyon cv2.imencode + ndarray.tofile kullanarak
    platformdan bagimsiz kaydetme saglar.

    Parametreler
    ------------
    path : str | Path
        Hedef dosya yolu. PNG, JPG veya JPEG uzantisi olmalidir;
        uzanti encode kalitesini/formatini belirler.
    img : np.ndarray
        Kaydedilecek goruntu dizisi (BGR ya da gri tonlamali).

    Dondurur
    --------
    None

    Basarisiz Olabilir
    ------------------
    - ValueError : Uzanti desteklenmiyorsa veya encode basarisiz olursa.
    - OSError    : Hedef dizin mevcut degilse veya yazma izni yoksa.
    """
    path = Path(path)

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Desteklenmeyen kayit uzantisi: '{path.suffix}'. "
            f"Desteklenen uzantilar: {_SUPPORTED_EXTENSIONS}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    success, buf = cv2.imencode(path.suffix.lower(), img)
    if not success:
        raise ValueError(
            f"cv2.imencode '{path.suffix}' formati icin basarisiz oldu."
        )

    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# Bilgi cikarma
# ---------------------------------------------------------------------------

def get_image_info(img: np.ndarray) -> dict:
    """Goruntuden temel istatistiksel ve yapisal bilgileri cikarir.

    Problem
    -------
    Bir goruntu dizisini hizlica karakterize etmek; boyut, renk derinligi,
    parlaklik ve kontrast gibi tanimlayici metrikleri tek sozlukte sunmak.

    Parametreler
    ------------
    img : np.ndarray
        BGR, gri tonlamali (2-D) ya da herhangi bir sayida kanalli goruntu
        dizisi. uint8 veya float turleri kabul edilir.

    Dondurur
    --------
    dict
        Anahtar-deger ciftleri:

        - width  (int)            : Goruntu genisligi (piksel)
        - height (int)            : Goruntu yuksekligi (piksel)
        - channels (int)          : Kanal sayisi (gri -> 1, BGR -> 3, BGRA -> 4)
        - dtype  (str)            : NumPy veri tipi adi (or. 'uint8')
        - min_pixel (float)       : En dusuk piksel degeri
        - max_pixel (float)       : En yuksek piksel degeri
        - mean_brightness (float) : Ortalama piksel parlakligi (gri kanalda)
        - std_contrast (float)    : Gri kanal standart sapmasi
        - range_contrast (float)  : Gri kanalda (max - min) degeri

    Uyari - float goruntular
    ------------------------
    float32/float64 dizilerde min/max/mean degerleri [0, 1] veya [0, 255]
    disina cikabilir; bu alanlar ham sayisal degeri yansitir, normalize
    edilmis bir aralik garantisi vermez.

    Basarisiz Olabilir
    ------------------
    - TypeError : img bir NumPy dizisi degilse.
    - Tamamen siyah/beyaz goruntulerde std_contrast = 0 doner; bu normal
      bir sonuctur, hata degildir.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img bir np.ndarray olmalidir, alinan: {type(img)}")

    if img.ndim == 2:
        height, width = img.shape
        channels = 1
        gray = img
    else:
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels >= 3 else img[:, :, 0]

    gray_float = gray.astype(np.float64)

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(img.dtype),
        "min_pixel": float(img.min()),
        "max_pixel": float(img.max()),
        "mean_brightness": float(gray_float.mean()),
        "std_contrast": float(gray_float.std()),
        "range_contrast": float(gray_float.max() - gray_float.min()),
    }
