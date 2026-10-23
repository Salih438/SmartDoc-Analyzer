"""
tests/test_noise.py
-------------------
src/evaluation/noise.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    simulate_uneven_illumination,
)


@pytest.fixture
def sample_gray() -> np.ndarray:
    return np.full((64, 64), 128, dtype=np.uint8)


def test_gaussian_noise_reproducibility(sample_gray: np.ndarray) -> None:
    n1 = add_gaussian_noise(sample_gray, sigma=20, seed=123)
    n2 = add_gaussian_noise(sample_gray, sigma=20, seed=123)
    np.testing.assert_array_equal(n1, n2)


def test_gaussian_noise_different_seed(sample_gray: np.ndarray) -> None:
    n1 = add_gaussian_noise(sample_gray, sigma=20, seed=123)
    n2 = add_gaussian_noise(sample_gray, sigma=20, seed=456)
    assert not np.array_equal(n1, n2)


def test_gaussian_noise_negative_sigma_raises(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma negatif olamaz"):
        add_gaussian_noise(sample_gray, sigma=-5.0)


def test_salt_pepper_noise_reproducibility(sample_gray: np.ndarray) -> None:
    n1 = add_salt_pepper_noise(sample_gray, amount=0.05, seed=42)
    n2 = add_salt_pepper_noise(sample_gray, amount=0.05, seed=42)
    np.testing.assert_array_equal(n1, n2)


def test_salt_pepper_noise_zero_amount(sample_gray: np.ndarray) -> None:
    n = add_salt_pepper_noise(sample_gray, amount=0.0)
    np.testing.assert_array_equal(n, sample_gray)


def test_salt_pepper_noise_invalid_amount(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="amount"):
        add_salt_pepper_noise(sample_gray, amount=1.5)


def test_simulate_uneven_illumination_reproducibility(sample_gray: np.ndarray) -> None:
    u1 = simulate_uneven_illumination(sample_gray, strength=0.5, seed=99)
    u2 = simulate_uneven_illumination(sample_gray, strength=0.5, seed=99)
    np.testing.assert_array_equal(u1, u2)


def test_simulate_uneven_illumination_zero_strength(sample_gray: np.ndarray) -> None:
    u = simulate_uneven_illumination(sample_gray, strength=0.0)
    np.testing.assert_array_equal(u, sample_gray)


def test_simulate_uneven_illumination_range(sample_gray: np.ndarray) -> None:
    u = simulate_uneven_illumination(sample_gray, strength=0.6, seed=10)
    assert u.dtype == np.uint8
    assert u.min() >= 0
    assert u.max() <= 255
    # Aydinlatma dengesizlestigi icin std sapma 0'dan buyuk olmali
    assert u.std() > 0.0


def test_simulate_uneven_illumination_invalid_strength(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strength"):
        simulate_uneven_illumination(sample_gray, strength=1.2)
