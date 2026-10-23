import numpy as np

from scripts.run_experiments import create_synthetic_document


def test_ground_truth_mask_is_supersampled_binary_and_compared_to_document() -> None:
    doc, mask = create_synthetic_document(width=1650, height=2200)
    assert set(np.unique(mask)).issubset({0, 255})
    doc_foreground = doc <= 127
    mask_foreground = mask > 0
    mismatch = int(np.count_nonzero(doc_foreground != mask_foreground))
    assert mismatch == int(np.count_nonzero(np.logical_xor(doc <= 127, mask > 0)))
    assert mismatch > 0
    assert mismatch / mask.size < 0.01
