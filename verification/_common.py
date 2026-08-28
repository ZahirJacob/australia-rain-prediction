"""Helpers shared by the verification scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_hashes_match(recorded: dict, current: dict, label: str) -> None:
    """Equality check whose failure names the artifact(s) whose hash differs."""
    keys = sorted(set(recorded) | set(current))
    mismatched = [k for k in keys if recorded.get(k) != current.get(k)]
    if mismatched:
        details = "; ".join(
            f"{k}: registrado={(recorded.get(k) or '-')[:16]} actual={(current.get(k) or '-')[:16]}"
            for k in mismatched
        )
        raise AssertionError(f"{label}: hash distinto para {mismatched} ({details})")
