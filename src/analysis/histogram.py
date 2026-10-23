"""
src/analysis/histogram.py
--------------------------
Gri tonlamali goruntular icin histogram hesaplama ve gorsellestirme modulu.

Not: matplotlib.use() cagrilmaz; bunun yerine Figure nesnesi dogrudan
olusturulur. Bu yaklasim Streamlit / Jupyter ortamlarinda global pyplot
durumunu bozmaz.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.figure
import matplotlib.axes


# ---------------------------------------------------------------------------
# Hesaplama
# ---------------------------------------------------------------------------

def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """Gri tonlamali bir goruntuun 256-bin piksel yogunluk histogramini hesaplar.

    Problem
    -------
    Bir gOruntunun ton dagilimini nicel olarak ifade etmek; iyilestirme
    algoritmalarinin oncesi/sonrasi karsilastirmasinda referans veri uretmek.

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi. uint8 beklenir; float goruntular
        otomatik olarak [0, 255] araligina kliplenir ve uint8'e donusturulur.

    Dondurur
    --------
    np.ndarray
        Sekil (256,), dtype float32. Her indeks [0..255] araligindaki
        piksel sayisini verir.

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse (RGB/BGR gecirilmisse).
    - Tamamen tekduze goruntulerde (or. np.zeros) yalnizca tek bin dolu
      olur; bu beklenen davranistir.
    """
    if gray.ndim != 2:
        raise ValueError(
            f"compute_histogram 2-D gri goruntu bekliyor, alinan sekil: {gray.shape}"
        )

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Gorsellestirme
# ---------------------------------------------------------------------------

def plot_histogram(
    gray: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Gri tonlamali goruntuun histogramini cizer ve Figure/Axes dondurur.

    Problem
    -------
    Histogram dagilimini gorsel olarak incelemek; Streamlit veya Jupyter
    not defteri gibi ortamlara kolayca gomülebilecek bir figur uretmek.

    Uygulama Notu
    -------------
    pyplot global durumuna bagimlilik yoktur. Yeni figur gerektiginde
    matplotlib.figure.Figure() dogrudan olusturulur; bu Streamlit
    ortaminda thread-safe calismayi destekler.

    Parametreler
    ------------
    gray : np.ndarray
        2-D gri tonlamali goruntu dizisi (compute_histogram ile ayni
        kisitlar gecerlidir).
    ax : matplotlib.axes.Axes | None, varsayilan None
        Varolan bir Axes nesnesine cizim yapmak icin gecirilebilir.
        None ise yeni Figure ve Axes olusturulur.

    Dondurur
    --------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        (fig, ax) cifti. ax=None iken yeni fig olusturulur; ax
        verildiginde fig = ax.get_figure() doner.

    Basarisiz Olabilir
    ------------------
    - ValueError : gray 2-D degilse (compute_histogram'dan yayilir).
    """
    hist = compute_histogram(gray)
    bins = np.arange(256)

    if ax is None:
        fig = matplotlib.figure.Figure(figsize=(7, 3), tight_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax.fill_between(bins, hist, alpha=0.6, color="#4a90d9", label="Piksel Yogunlugu")
    ax.plot(bins, hist, color="#2c5f8a", linewidth=0.8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Piksel Degeri (0-255)", fontsize=9)
    ax.set_ylabel("Piksel Sayisi", fontsize=9)
    ax.set_title("Gri Tonlama Histogrami", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    return fig, ax