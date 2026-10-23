"""Ensure Python sources are UTF-8 and contain no damaged Turkish text."""

import ast
from pathlib import Path
import re
import tokenize


BROKEN_LETTER_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]\?[A-Za-zÀ-ÖØ-öø-ÿ]")
QUESTION_PREFIX_PATTERN = re.compile(r"(?:^|[\s([{\"'])\?[a-zçğıöşü]")
REGEX_WHITELIST = {
    "test_source_encoding.py": (
        "BROKEN_LETTER_PATTERN = re.compile(",
        "QUESTION_PREFIX_PATTERN = re.compile(",
    )
}
REAL_QUESTION_MARK_WHITELIST = {
    "test_source_encoding.py": ('assert QUESTION_PREFIX_PATTERN.search("Bu doğru mu?")',)
}


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Return line numbers belonging to module/class/function docstrings."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def test_python_sources_are_utf8_without_replacement_or_broken_letters() -> None:
    root = Path(__file__).parents[1]
    for folder in ("src", "tests", "scripts"):
        for path in (root / folder).rglob("*.py"):
            raw = path.read_bytes()
            assert not raw.startswith(b"\xef\xbb\xbf"), path
            text = raw.decode("utf-8", errors="strict")
            assert "\ufffd" not in text, path
            tree = ast.parse(text, filename=str(path))
            protected_lines = _docstring_lines(tree)
            protected_lines.update(
                token.start[0]
                for token in tokenize.generate_tokens(iter(text.splitlines(keepends=True)).__next__)
                if token.type == tokenize.COMMENT
            )
            for line_number, line in enumerate(text.splitlines(), start=1):
                allowed = any(line.lstrip().startswith(prefix) for prefix in REGEX_WHITELIST.get(path.name, ()))
                legitimate_question = any(
                    prefix in line for prefix in REAL_QUESTION_MARK_WHITELIST.get(path.name, ())
                )
                if not allowed:
                    assert BROKEN_LETTER_PATTERN.search(line) is None, f"{path}:{line_number}: {line}"
                if not allowed and not legitimate_question:
                    assert QUESTION_PREFIX_PATTERN.search(line) is None, f"{path}:{line_number}: {line}"


def test_question_prefix_detector_catches_leading_corruption_only() -> None:
    assert QUESTION_PREFIX_PATTERN.search("?" + "şüpheli")
    assert QUESTION_PREFIX_PATTERN.search("Bu doğru mu?") is None
