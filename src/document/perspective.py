"""Belge dörtgenini sıralama ve perspektif düzeltme yardımcıları."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Dört noktayı açı sırasıyla ``TL, TR, BR, BL`` olarak döndürür.

    Problem: x+y / y-x sezgisi 45 derece civarında çakışabilir. Algoritma:
    merkez etrafında açı sıralaması yapar, ardından en üst-sol köşeyi başa
    alıp yönü saat yönüne çevirir. Dört nokta değilse başarısız olur.
    """
    points = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2):
        raise ValueError("order_corners: tam olarak dört (x, y) noktası bekleniyor")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # atan2 sırası saat yönünün tersidir; dörtgeni saat yönüne çevir.
    if cv2.contourArea(ordered.reshape(-1, 1, 2), oriented=True) < 0:
        ordered = ordered[::-1]
    # Choose the visually upper-left origin by minimum x+y. At diagonal ties,
    # topmost then leftmost is deterministic and independent of input order.
    sums = ordered[:, 0] + ordered[:, 1]
    minimum = float(np.min(sums))
    candidates = np.flatnonzero(np.isclose(sums, minimum, atol=1e-4))
    start = int(candidates[np.lexsort((ordered[candidates, 0], ordered[candidates, 1]))[0]])
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def compute_output_size(corners: np.ndarray) -> tuple[int, int]:
    """Karşılıklı kenarların büyüklerini kullanarak (genişlik, yükseklik) döndürür."""
    p = order_corners(corners)
    width = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))
    height = max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))
    return max(1, int(round(width))), max(1, int(round(height)))


def _validate_quad(corners: np.ndarray) -> np.ndarray:
    p = order_corners(corners)
    area = abs(float(cv2.contourArea(p.reshape(-1, 1, 2))))
    cross = [
        float(np.linalg.det(np.stack([p[(i + 1) % 4] - p[i], p[(i + 2) % 4] - p[(i + 1) % 4]])))
        for i in range(4)
    ]
    if area < 1e-3 or len(set(map(tuple, p))) != 4 or not (all(x > 0 for x in cross) or all(x < 0 for x in cross)):
        raise ValueError("four_point_transform: dejenere veya konveks olmayan dörtgen")
    return p


def four_point_transform(
    img: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Belgeyi ``getPerspectiveTransform`` ve ``warpPerspective`` ile düzeltir.

    Başarısız olur: çakışan, sıfır alanlı veya konveks olmayan köşelerde
    ``ValueError``; OpenCV'nin desteklemediği görüntü tiplerinde OpenCV hatası.
    """
    p = _validate_quad(corners)
    w, h = output_size or compute_output_size(p)
    if w < 1 or h < 1:
        raise ValueError("four_point_transform: çıktı boyutu pozitif olmalı")
    destination = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(p, destination)
    return cv2.warpPerspective(img, matrix, (w, h)), matrix
