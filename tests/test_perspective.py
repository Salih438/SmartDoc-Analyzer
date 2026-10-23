import cv2
import numpy as np
import pytest

from src.document.perspective import compute_output_size, four_point_transform, order_corners


def test_order_corners_rotations_and_permutations() -> None:
    for angle in (0, 30, 45, 90, 135):
        base = np.array([[20, 20], [180, 20], [180, 120], [20, 120]], np.float32)
        matrix = cv2.getRotationMatrix2D((100, 70), angle, 1.0)
        points = cv2.transform(base[None], matrix)[0]
        expected = order_corners(points)
        assert expected.shape == (4, 2)
        assert np.allclose(order_corners(points[[2, 0, 3, 1]]), expected)
        assert np.allclose(order_corners(points[::-1]), expected)


def test_output_size_and_transform() -> None:
    corners = np.array([[0, 0], [99, 0], [99, 49], [0, 49]], np.float32)
    assert compute_output_size(corners) == (99, 49)
    image = np.zeros((50, 100), np.uint8)
    warped, matrix = four_point_transform(image, corners)
    assert warped.shape == (49, 99)
    assert matrix.shape == (3, 3)


@pytest.mark.parametrize("points", [np.zeros((4, 2)), [[0, 0], [1, 0], [0, 0], [1, 1]], [[0, 0], [2, 0], [1, 1], [0, 2]]])
def test_degenerate_quads_raise(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        four_point_transform(np.zeros((10, 10), np.uint8), np.asarray(points, np.float32))
