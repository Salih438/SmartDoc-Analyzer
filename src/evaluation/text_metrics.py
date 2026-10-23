"""Metin normalleştirme ve karakter/kelime hata ölçümleri."""

from __future__ import annotations

import unicodedata


def normalize_text(text: str, case_sensitive: bool = True) -> str:
    """NFC ve boşluk normalleştir; isteğe bağlı Türkçe uyumlu harf eşle."""
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if case_sensitive:
        return normalized
    # Turkish casing is explicit: Python lower() maps I to i, not dotless ı.
    return normalized.translate(str.maketrans({"I": "ı", "İ": "i"})).casefold()


def levenshtein(reference: str, hypothesis: str) -> int:
    """Return minimum insertion/deletion/substitution count (two-row DP)."""
    return _sequence_distance(tuple(reference), tuple(hypothesis))


def _sequence_distance(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, start=1):
        current = [row]
        for column, right in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str, case_sensitive: bool = True) -> float:
    """Character error rate after NFC/whitespace normalization."""
    ref = normalize_text(reference, case_sensitive)
    hyp = normalize_text(hypothesis, case_sensitive)
    return levenshtein(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str, case_sensitive: bool = True) -> float:
    """Word error rate after NFC/whitespace normalization."""
    ref = normalize_text(reference, case_sensitive).split()
    hyp = normalize_text(hypothesis, case_sensitive).split()
    return _sequence_distance(tuple(ref), tuple(hyp)) / max(1, len(ref))
