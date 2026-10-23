"""Klasik OpenCV ile belge sınırı adaylarını bulma."""

from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np

from src.document.perspective import order_corners
from src.segmentation.edges import auto_canny

AREA_WEIGHT = 0.35
FILL_WEIGHT = 0.20
ANGLE_WEIGHT = 0.20
EDGE_WEIGHT = 0.15
BORDER_PENALTY = 0.50
MIN_SCORE = 0.67
MAX_CONFIDENT_AREA_RATIO = 0.75


@dataclass
class DetectionResult:
    """Algılama sonucu, güven durumu ve hata ayıklama ara görüntüleri."""

    corners: np.ndarray | None
    status: str
    score: float
    reason: str
    strategy: str
    debug: dict[str, object] = field(default_factory=dict)


def _angles_score(p: np.ndarray) -> float:
    values = []
    for i in range(4):
        a = p[(i - 1) % 4] - p[i]
        b = p[(i + 1) % 4] - p[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        values.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return float(np.mean([max(0.0, 1.0 - abs(x - 90.0) / 30.0) for x in values]))


def _candidate_score(
    p: np.ndarray, contour: np.ndarray, image_shape: tuple[int, int]
) -> tuple[float, str, dict[str, float]]:
    h, w = image_shape
    area_ratio = cv2.contourArea(p.reshape(-1, 1, 2)) / (w * h)
    contour_area = cv2.contourArea(contour)
    fill = min(1.0, area_ratio * w * h / max(contour_area, 1.0))
    edge_ratios = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    opposite = min(edge_ratios[0], edge_ratios[2]) / max(edge_ratios[0], edge_ratios[2], 1e-6)
    opposite *= min(edge_ratios[1], edge_ratios[3]) / max(edge_ratios[1], edge_ratios[3], 1e-6)
    border = float(np.mean([(x <= 2 or y <= 2 or x >= w - 3 or y >= h - 3) for x, y in p]))
    area_component = AREA_WEIGHT * min(1.0, area_ratio / 0.8)
    fill_component = FILL_WEIGHT * fill
    angle_component = ANGLE_WEIGHT * _angles_score(p)
    opposite_component = EDGE_WEIGHT * opposite
    border_component = BORDER_PENALTY * border
    score = area_component + fill_component + angle_component + opposite_component - border_component
    reason = "görüntü sınırına yapışıyor" if border >= 0.5 else "geometrik skor yeterli"
    return float(score), reason, {
        "area_ratio": float(area_ratio), "fill": float(fill),
        "angle_score": float(_angles_score(p)), "opposite_edge_score": float(opposite),
        "border_fraction": float(border), "area_weighted": float(area_component),
        "fill_weighted": float(fill_component), "angle_weighted": float(angle_component),
        "opposite_weighted": float(opposite_component), "border_penalty": float(border_component),
    }


def detect_document(
    img: np.ndarray,
    strategy: str = "canny",
    canny_mode: str = "fixed",
    work_height: int = 800,
) -> DetectionResult:
    """Belge sınırını Canny veya açık kâğıt/koyu arka plan eşiklemesiyle bulur.

    Algoritma çalışma boyutuna küçültür, griye çevirir, hafif blur ve kapanma
    uygular; konturları alan, konveks kılıf ve ``approxPolyDP`` epsilon
    aralığıyla dörtlere indirger. Threshold stratejisi açık kâğıt/koyu masa
    varsayar. Çok küçük, konveks olmayan, düşük kontrastlı veya kadrajdan
    taşan belgelerde başarısız olabilir; tüm görüntüyü fallback saymaz.
    """
    if strategy not in {"canny", "threshold"}:
        raise ValueError("strategy 'canny' veya 'threshold' olmalı")
    if canny_mode not in {"fixed", "auto", "otsu"}:
        raise ValueError("canny_mode 'fixed', 'auto' veya 'otsu' olmalı")
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        dtype = getattr(img, "dtype", type(img).__name__)
        raise TypeError(f"detect_document: uint8 görüntü bekleniyor, alınan dtype: {dtype}")
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] in {3, 4}:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("detect_document: gri veya BGR/BGRA görüntü bekleniyor")
    scale = min(1.0, work_height / gray.shape[0])
    work = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale))) if scale < 1 else gray.copy()
    if int(work.max()) == int(work.min()):
        empty = np.zeros_like(work)
        return DetectionResult(
            None,
            "not_found",
            0.0,
            "kontur_yok: tekdüze görüntüde sınır kontrastı bulunamadı",
            strategy,
            {"edges": empty, "mask": empty.copy(), "candidates": [], "scale": scale,
             "stages": {"edge_pixels": 0, "closing_pixels": 0, "contours": 0,
                        "area_pass": 0, "hulls": 0, "approx_attempts": 0,
                        "approx_four": 0, "score_pass": 0, "score_reject": 0,
                        "approx_vertices": []}},
        )
    smooth = cv2.GaussianBlur(work, (5, 5), 0)
    if strategy == "canny":
        if canny_mode == "fixed":
            edges = cv2.Canny(smooth, 50, 150)
        elif canny_mode == "auto":
            edges = auto_canny(smooth)
        else:
            otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            edges = cv2.Canny(smooth, otsu * 0.5, otsu)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    else:
        _, mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        edges = cv2.Canny(smooth, 50, 150)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, str, dict[str, float]]] = []
    min_area = 0.10 * work.shape[0] * work.shape[1]
    stage = {
        "edge_pixels": int(np.count_nonzero(edges)),
        "closing_pixels": int(np.count_nonzero(mask)),
        "contours": len(contours),
        "area_pass": 0,
        "hulls": 0,
        "approx_attempts": 0,
        "approx_four": 0,
        "score_pass": 0,
        "score_reject": 0,
    }
    approx_vertices: list[int] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        stage["area_pass"] += 1
        hull = cv2.convexHull(contour)
        stage["hulls"] += 1
        perimeter = cv2.arcLength(hull, True)
        for epsilon_factor in np.linspace(0.01, 0.06, 11):
            approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
            stage["approx_attempts"] += 1
            approx_vertices.append(len(approx))
            if len(approx) == 4 and cv2.isContourConvex(approx):
                stage["approx_four"] += 1
                p = order_corners(approx.reshape(4, 2))
                score, reason, components = _candidate_score(p, contour, work.shape)
                candidates.append((score, p, reason, components))
                break
    best_candidate_score = max((candidate[0] for candidate in candidates), default=0.0)
    stage["score_pass"] = sum(candidate[0] >= MIN_SCORE for candidate in candidates)
    stage["score_reject"] = len(candidates) - stage["score_pass"]
    stage["approx_vertices"] = approx_vertices  # type: ignore[assignment]
    if not candidates:
        if not contours:
            reason = "kontur_yok"
        elif stage["area_pass"] == 0:
            reason = "alan_filtresi"
        elif stage["approx_four"] == 0:
            reason = "approx_4_kose_yok"
        else:
            reason = "aday_yok"
        return DetectionResult(None, "not_found", 0.0, reason, strategy, {"edges": edges, "mask": mask, "candidates": [], "stages": stage})
    score, p, reason, components = max(candidates, key=lambda item: item[0])
    original = p / scale
    status = "ok" if score >= MIN_SCORE else "low_confidence"
    if status != "ok":
        reason = f"skor eşik altında ({score:.3f}); {reason}"
        reason = "skor_esik_alti: " + reason
    if components["area_ratio"] > MAX_CONFIDENT_AREA_RATIO:
        status = "low_confidence"
        reason = (
            f"aday alanı görüntünün %{components['area_ratio'] * 100:.1f}'ini kaplıyor; "
            f"güven sınırı %{MAX_CONFIDENT_AREA_RATIO * 100:.0f}"
        )
    return DetectionResult(
        original.astype(np.float32), status, score, reason, strategy,
        {"edges": edges, "mask": mask, "candidates": [x[1] for x in candidates],
         "candidate_scores": [
             {"score": item[0], "components": item[3]} for item in candidates
         ], "selected_score_components": components, "scale": scale, "stages": stage},
    )
