from __future__ import annotations

import pytest

from src.evaluation.text_metrics import cer, levenshtein, normalize_text, wer


def test_known_levenshtein_cer_wer_values() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert cer("kitten", "sitting") == pytest.approx(0.5)
    assert wer("one two", "one too") == pytest.approx(0.5)


def test_normalize_nfc_whitespace_and_case_sensitive_default() -> None:
    assert normalize_text("Cafe\u0301   test") == "Café test"
    assert normalize_text("İ I ı i") == "İ I ı i"
    assert normalize_text("İ I ı i", case_sensitive=False) == "i ı ı i"


def test_empty_reference_rates_are_finite() -> None:
    assert cer("", "new") == 3.0
    assert wer("", "new words") == 2.0
