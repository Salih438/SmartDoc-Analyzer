from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation.scenes import SCENARIOS, make_scene
from src.analysis.image_info import save_image
from src.pipeline import PipelineConfig, process_file, process_image


def _document() -> np.ndarray:
    image = np.full((500, 360, 3), 245, dtype=np.uint8)
    cv2.putText(image, "SMARTDOC", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
    return image


def test_pipeline_frontal_returns_binary_and_preserves_input() -> None:
    scene, _ = make_scene(_document(), "frontal_easy", seed=12)
    original = scene.copy()
    result = process_image(scene, PipelineConfig())
    assert result.detection.status == "ok"
    assert result.rectified is True
    assert result.binary is not None
    assert set(np.unique(result.binary)).issubset({0, 255})
    assert np.array_equal(scene, original)


def test_not_found_warns_and_does_not_warp_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((100, 120, 3), 127, dtype=np.uint8)
    import src.pipeline as pipeline_module

    def fail_if_warped(*args: object, **kwargs: object) -> None:
        raise AssertionError("warp must not run after a rejected detection")

    monkeypatch.setattr(pipeline_module, "four_point_transform", fail_if_warped)
    result = process_image(image, PipelineConfig())
    assert result.rectified is False
    assert result.detection.status == "not_found"
    assert result.warnings
    assert result.gray is None


def test_allow_unrectified_continues_with_warning() -> None:
    image = np.full((100, 120), 127, dtype=np.uint8)
    result = process_image(image, PipelineConfig(allow_unrectified=True))
    assert result.rectified is False
    assert result.warnings
    assert result.gray is not None


def test_denoise_precedes_binarization() -> None:
    scene, _ = make_scene(_document(), "frontal_easy", seed=4)
    result = process_image(scene, PipelineConfig(denoise="median", binarize="otsu"))
    names = [step["ad"] for step in result.steps]
    assert names.index("denoise") < names.index("binarize")


def test_optional_detection_median_precedes_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.pipeline as pipeline_module

    captured: dict[str, np.ndarray] = {}

    def fake_detector(image: np.ndarray, **kwargs: object):
        captured["image"] = image.copy()
        return pipeline_module.DetectionResult(None, "not_found", 0.0, "test", "canny")

    monkeypatch.setattr(pipeline_module, "detect_document", fake_detector)
    image = _document()
    result = process_image(image, PipelineConfig(detect_denoise=True))
    assert result.detection.status == "not_found"
    assert "detect_denoise" in [step["ad"] for step in result.steps]
    assert captured["image"].ndim == 2


def test_invalid_pipeline_config_raises() -> None:
    with pytest.raises(ValueError, match="enhance"):
        PipelineConfig(enhance="histogram")  # type: ignore[arg-type]


def test_scaled_kernel_changes_with_resolution() -> None:
    small = np.full((450, 330, 3), 255, dtype=np.uint8)
    large = np.full((2200, 1650, 3), 255, dtype=np.uint8)
    small_result = process_image(
        small, PipelineConfig(detect=False, allow_unrectified=True, denoise="median", binarize="adaptive_mean")
    )
    large_result = process_image(
        large, PipelineConfig(detect=False, allow_unrectified=True, denoise="median", binarize="adaptive_mean")
    )
    small_kernel = next(step["parametreler"]["ksize"] for step in small_result.steps if step["ad"] == "denoise")
    large_kernel = next(step["parametreler"]["ksize"] for step in large_result.steps if step["ad"] == "denoise")
    assert small_kernel < large_kernel
    assert small_kernel % 2 == large_kernel % 2 == 1
    small_block = next(step["parametreler"]["block_size"] for step in small_result.steps if step["ad"] == "binarize")
    large_block = next(step["parametreler"]["block_size"] for step in large_result.steps if step["ad"] == "binarize")
    assert small_block < large_block
    assert small_block % 2 == large_block % 2 == 1


def test_auto_rules_use_synthetic_brightness_contrast_and_noise() -> None:
    from src.pipeline import _enhance_auto

    low_contrast = np.full((120, 160), 140, dtype=np.uint8)
    _, selected, _ = _enhance_auto(low_contrast)
    assert selected == "stretch"

    rng = np.random.default_rng(24)
    noisy = np.clip(128 + rng.normal(0, 12, (120, 160)), 0, 255).astype(np.uint8)
    result = process_image(
        noisy,
        PipelineConfig(detect=False, allow_unrectified=True, enhance="auto", denoise="auto", binarize="none"),
    )
    denoise_step = next(step for step in result.steps if step["ad"] == "denoise")
    assert denoise_step["parametreler"]["method"] == "median+bilateral"
    assert "noise_sigma_estimate" in denoise_step["parametreler"]


def test_process_file_supports_unicode_directory(tmp_path: Path) -> None:
    folder = tmp_path / "Çalışma_Örnek_Şube"
    folder.mkdir()
    input_path = folder / "belge_şüphe.png"
    save_image(input_path, make_scene(_document(), "frontal_easy", seed=12)[0])
    output_dir = folder / "çıktı"
    result = process_file(input_path, output_dir, PipelineConfig())
    assert result.detection.status == "ok"
    assert list(output_dir.glob("*.json"))
    assert list(output_dir.glob("*binary.png"))


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "partial_crop"])
def test_scene_ground_truth_inside_frame_except_partial_crop(scenario: str) -> None:
    scene, corners = make_scene(_document(), scenario, seed=3)
    h, w = scene.shape[:2]
    assert np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < w)
    assert np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < h)


def test_reverse_polarity_is_not_silently_marked_ok_when_wrong() -> None:
    image = np.full((420, 320), 235, dtype=np.uint8)
    gt = np.array([[35, 45], [285, 45], [285, 375], [35, 375]], dtype=np.float32)
    cv2.fillPoly(image, [gt.astype(np.int32)], 38)
    from src.document.detection import detect_document

    result = detect_document(image, strategy="threshold")
    if result.corners is None or result.status != "ok":
        assert result.status in {"low_confidence", "not_found"}
    else:
        from src.evaluation.metrics import corner_error, quad_iou

        _, error_ratio = corner_error(gt, result.corners)
        assert error_ratio < 0.02 and quad_iou(gt, result.corners, image.shape) > 0.90


def test_config_sidecar_is_valid_json(tmp_path: Path) -> None:
    folder = tmp_path / "Şema"
    folder.mkdir()
    path = folder / "input.png"
    save_image(path, make_scene(_document(), "frontal_easy", seed=12)[0])
    out = folder / "out"
    process_file(path, out, PipelineConfig())
    metadata = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert "config" in metadata and "steps" in metadata
