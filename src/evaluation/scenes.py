"""Tohumlu, sentetik perspektif belge sahneleri ve kesin GT köşeleri."""

from __future__ import annotations

import cv2
import numpy as np

from src.document.perspective import order_corners


SCENARIOS = ("frontal_easy", "frontal_clean", "mild_tilt", "strong_perspective", "rotated_30", "rotated_45", "rotated_90", "low_contrast_bg", "low_contrast_hard", "low_contrast_hard_noisy", "dark_bg", "clutter", "partial_crop", "noisy_shadow")


def make_scene(doc: np.ndarray, scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Belgeyi dokulu/gradyanlı renkli arka plana homografiyle yerleştirir.

    GT köşeleri, kullanılan hedef dörtgenin kendisidir; bu nedenle piksel
    koordinatları yaklaşık değil homografiden üretilir. Taşan belge ve
    ``clutter`` senaryosu algılamanın doğal başarısızlık örnekleridir.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"bilinmeyen senaryo: {scenario}")
    rng = np.random.default_rng(seed)
    source = doc if doc.ndim == 3 else cv2.cvtColor(doc, cv2.COLOR_GRAY2BGR)
    sh, sw = source.shape[:2]
    h, w = 900, 700
    yy, xx = np.mgrid[:h, :w]
    base = 100 + (xx / w * 35 + yy / h * 20).astype(np.float32)
    noise = rng.normal(0, 8, (h, w)).astype(np.float32)
    scene_kind = "low_contrast_hard" if scenario == "low_contrast_hard_noisy" else scenario
    bg_value = 205 if scene_kind in {"low_contrast_bg", "clutter", "low_contrast_hard"} else (35 if scene_kind == "dark_bg" else base + noise)
    if scene_kind == "frontal_clean":
        bg_value = np.full((h, w), 120, dtype=np.uint8)
    if scene_kind in {"low_contrast_bg", "clutter"}:
        source = np.where(source.astype(np.float32).mean(axis=2, keepdims=True) > 200, 235, source).astype(np.uint8)
    elif scene_kind == "low_contrast_hard":
        source = np.where(
            source.astype(np.float32).mean(axis=2, keepdims=True) > 200,
            215,
            source,
        ).astype(np.uint8)
        bg_value = np.full((h, w), 205, dtype=np.uint8)
    elif scene_kind == "frontal_clean":
        source = np.where(source.astype(np.float32).mean(axis=2, keepdims=True) > 200, 245, source).astype(np.uint8)
    bg = np.clip(bg_value, 0, 255).astype(np.uint8)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    margin_x, margin_y = 90, 70
    target = np.array([[margin_x, margin_y], [w - margin_x, margin_y], [w - margin_x, h - margin_y], [margin_x, h - margin_y]], dtype=np.float32)
    if scene_kind == "mild_tilt":
        target += np.array([[15, 10], [-10, 0], [10, -15], [-10, 5]], np.float32)
    elif scene_kind == "strong_perspective":
        target = np.array([[170, 120], [570, 80], [630, 820], [80, 850]], dtype=np.float32)
    elif scene_kind in {"rotated_30", "rotated_45", "rotated_90"}:
        angle = int(scene_kind.split("_")[1])
        center = np.array([w / 2, h / 2], np.float32)
        rot = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        target = cv2.transform(target[None], rot)[0]
        bounds_min = target.min(axis=0)
        bounds_max = target.max(axis=0)
        extent = bounds_max - bounds_min
        fit_scale = min((w - 80) / extent[0], (h - 80) / extent[1], 1.0)
        target = (target - (bounds_min + bounds_max) / 2) * fit_scale + center
    elif scene_kind == "partial_crop":
        target += np.array([[-120, -100], [30, -70], [60, 60], [-130, 40]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32), target)
    # Replicate paper pixels at the source boundary so black warp padding does
    # not create an artificial high-contrast outline around the page.
    warped = cv2.warpPerspective(source, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    occupied = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), matrix, (w, h))
    scene = np.where(occupied[..., None] > 0, warped, bg)
    if scene_kind == "clutter":
        # Rectangle area is 501,600 px, larger than the nominal page quadrilateral.
        cv2.rectangle(scene, (20, 100), (680, 860), (225, 225, 225), 5)
    if scene_kind == "noisy_shadow":
        shadow = np.zeros((h, w), np.uint8)
        cv2.ellipse(shadow, (w // 2, h // 2), (260, 380), -25, 0, 360, 90, -1)
        scene = np.clip(scene.astype(np.int16) - shadow[..., None] // 3, 0, 255).astype(np.uint8)
    if scenario == "low_contrast_hard_noisy":
        scene = np.clip(
            scene.astype(np.float32) + rng.normal(0, 10.0, scene.shape), 0, 255
        ).astype(np.uint8)
    return scene.astype(np.uint8), order_corners(target)
