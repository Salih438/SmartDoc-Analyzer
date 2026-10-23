from __future__ import annotations

import numpy as np

from scripts.run_enhancement_experiments import (
    METHOD_NAMES,
    _all_outputs,
    _condition_input,
    _gamma_paper,
    _make_document,
)


def test_enhancement_experiment_conditions_are_seeded_and_change_exposure() -> None:
    clean = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (64, 1))
    dark_a = _condition_input(clean, "karanlik", 11)
    dark_b = _condition_input(clean, "karanlik", 11)
    bright = _condition_input(clean, "asiri_parlak", 11)

    assert np.array_equal(dark_a, dark_b)
    assert not np.array_equal(dark_a, clean)
    assert float(bright.mean()) > float(clean.mean())


def test_experiment_has_multiple_distinct_document_designs() -> None:
    page_a, mask_a = _make_document("belge_a")
    page_b, mask_b = _make_document("belge_b")
    page_c, mask_c = _make_document("belge_c")

    assert len({page_a.tobytes(), page_b.tobytes(), page_c.tobytes()}) == 3
    assert len({mask_a.tobytes(), mask_b.tobytes(), mask_c.tobytes()}) == 3


def test_gamma_paper_moves_ninetieth_percentile_toward_target() -> None:
    gray = np.tile(np.linspace(50, 180, 256, dtype=np.uint8), (64, 1))
    output, gamma = _gamma_paper(gray)

    assert gamma > 1
    assert abs(float(np.percentile(output, 90)) - 0.9 * 255) <= 2


def test_experiment_threshold_methods_produce_binary_masks() -> None:
    gray = np.full((120, 160), 230, dtype=np.uint8)
    gray[25:95:10, 20:140] = 30
    _, outputs = _all_outputs(gray)

    assert set(outputs) == set(METHOD_NAMES)
    assert set(np.unique(outputs["sauvola"])) <= {0, 255}
    assert set(np.unique(outputs["niblack"])) <= {0, 255}
    assert set(np.unique(outputs["flatten+clahe+otsu"])) <= {0, 255}
