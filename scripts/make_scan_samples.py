"""Tarama modlarının sentetik örnek PNG'lerini üretir."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.image_info import save_image
from src.evaluation.noise import simulate_uneven_illumination
from src.evaluation.scenes import make_scene
from src.pipeline import PipelineConfig, process_image
from src.scan import SCAN_MODES, make_scan


SCENARIO_MAP = {
    "normal": "frontal_easy",
    "dengesiz_aydinlatma": "frontal_clean",
    "guclu_perspektif": "strong_perspective",
}


def _sample_document() -> np.ndarray:
    """Türkçe glifli kısa, renkli olmayan sentetik belge oluşturur."""
    page = np.full((700, 520, 3), 250, dtype=np.uint8)
    cv2.putText(page, "SMARTDOC", (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(page, "SCAN SAMPLE", (45, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(page, "12345 / 67890", (45, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.rectangle(page, (35, 90), (485, 440), (210, 210, 210), 2)
    return page


def generate_samples(output_dir: str | Path | None = None) -> list[Path]:
    """Üç sentetik sahnenin her tarama modunu tespit/warp sonrası PNG kaydeder."""
    target_dir = Path(output_dir) if output_dir is not None else ROOT / "results" / "scan_samples"
    target_dir.mkdir(parents=True, exist_ok=True)
    document = _sample_document()
    generated: list[Path] = []

    for index, (scene_name, scenario) in enumerate(SCENARIO_MAP.items()):
        scene, _ = make_scene(document, scenario, seed=71 + index)
        if scene_name == "dengesiz_aydinlatma":
            hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = simulate_uneven_illumination(hsv[:, :, 2], strength=0.7, seed=81)
            scene = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = process_image(scene, PipelineConfig(binarize="none"))
        if not result.rectified or result.rectified_image is None:
            raise RuntimeError(
                f"Örnek sahnede belge düzeltilemedi: {scene_name} "
                f"({result.detection.status}: {result.detection.reason})"
            )
        for mode in SCAN_MODES:
            output_path = target_dir / f"{scene_name}_{mode}.png"
            save_image(output_path, make_scan(result.rectified_image, mode))
            generated.append(output_path)
    return generated


def main() -> None:
    generated = generate_samples()
    print(f"{len(generated)} örnek PNG üretildi: {ROOT / 'results' / 'scan_samples'}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
