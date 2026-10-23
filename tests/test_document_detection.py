import numpy as np
import cv2
import pytest

from src.document.detection import detect_document
from src.evaluation.metrics import corner_error, quad_iou
from src.evaluation.scenes import make_scene


def test_scene_seed_reproducibility() -> None:
    doc = np.full((300, 220, 3), 245, np.uint8)
    first = make_scene(doc, "frontal_easy", 7)
    second = make_scene(doc, "frontal_easy", 7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_corner_error_is_circular_and_iou_basics() -> None:
    gt = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    pred = np.roll(gt, 2, axis=0)
    assert corner_error(gt, pred) == (0.0, 0.0)
    assert quad_iou(gt, gt, (20, 20)) == 1.0
    assert quad_iou(gt, gt + 20, (50, 50)) == 0.0


def test_detection_experiment_corner_distances_align_circular_order() -> None:
    from scripts.run_detection_experiments import _corner_distances

    gt = np.array([[10, 10], [80, 10], [80, 90], [10, 90]], dtype=np.float32)
    distances, mean = _corner_distances(gt, np.roll(gt, 1, axis=0))
    assert mean == 0.0
    assert distances == [0.0, 0.0, 0.0, 0.0]


def test_exterior_edge_gradient_is_not_inflated_by_interior_text() -> None:
    from scripts.run_detection_experiments import _boundary_measurements

    gt = np.array([[15, 15], [85, 15], [85, 85], [15, 85]], dtype=np.float32)
    plain = np.full((100, 100), 205, dtype=np.uint8)
    cv2.fillPoly(plain, [gt.astype(np.int32)], 215)
    with_text = plain.copy()
    cv2.putText(with_text, "TEXT", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 20, 2)
    edges = np.zeros_like(plain)
    without_text = _boundary_measurements(plain, edges, gt)
    with_text_metrics = _boundary_measurements(with_text, edges, gt)
    assert with_text_metrics["boundary_sobel_p90"] == without_text["boundary_sobel_p90"]
    assert with_text_metrics["boundary_sobel_max"] == without_text["boundary_sobel_max"]


def test_synthetic_detector_label_does_not_touch_paper_boundary() -> None:
    from scripts.run_detection_experiments import _detection_document, _boundary_measurements

    doc = _detection_document()
    scene, corners = make_scene(doc, "low_contrast_hard", seed=0)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    edge_stats = _boundary_measurements(gray, np.zeros_like(gray), corners)
    assert edge_stats["boundary_sobel_p90"] < 100


def test_uniform_image_is_not_found() -> None:
    result = detect_document(np.full((400, 300), 128, np.uint8))
    assert result.status == "not_found"
    assert result.corners is None


def test_uniform_threshold_image_is_not_ok() -> None:
    image = np.full((120, 160), 128, dtype=np.uint8)
    result = detect_document(image, strategy="threshold")
    assert result.status != "ok"


def test_otsu_canny_passes_ordered_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.document.detection as detection_module

    seen: list[tuple[float, float]] = []
    original_canny = cv2.Canny

    def checked_canny(image: np.ndarray, low: float, high: float, *args: object, **kwargs: object) -> np.ndarray:
        seen.append((low, high))
        return original_canny(image, low, high, *args, **kwargs)

    monkeypatch.setattr(detection_module.cv2, "Canny", checked_canny)
    detect_document(np.tile(np.arange(256, dtype=np.uint8), (100, 1)), canny_mode="otsu")
    assert seen and all(low <= high for low, high in seen)


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.uint16])
def test_detector_rejects_non_uint8(dtype: type) -> None:
    with pytest.raises(TypeError, match="uint8 görüntü bekleniyor"):
        detect_document(np.zeros((100, 100), dtype=dtype))


def test_document_touching_frame_is_reported_without_crash() -> None:
    image = np.full((300, 240), 80, np.uint8)
    image[0:300, 15:225] = 240
    result = detect_document(image, strategy="canny")
    assert result.status in {"ok", "low_confidence", "not_found"}


def test_same_brightness_paper_and_background_is_not_found() -> None:
    image = np.full((300, 240), 230, np.uint8)
    result = detect_document(image, strategy="threshold")
    assert result.status == "not_found"


def test_small_document_and_dark_polarity_are_executed() -> None:
    small = np.full((400, 300), 80, np.uint8)
    small[170:230, 120:180] = 240
    assert detect_document(small, strategy="canny").status == "not_found"

    dark_paper = np.full((400, 300), 230, np.uint8)
    dark_paper[50:350, 40:260] = 40
    assert detect_document(dark_paper, strategy="threshold").status in {"ok", "low_confidence", "not_found"}
