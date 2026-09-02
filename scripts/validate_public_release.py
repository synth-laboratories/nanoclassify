#!/usr/bin/env python3
"""Fail when the tracked public tree contains secrets or per-example answer material."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {".env", "predictions.jsonl", "errors.json"}
SECRET_PATTERNS = (
    re.compile(r"tinker_[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?m)^TINKER_API_KEY\s*=\s*[^\s#]+"),
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / value.decode() for value in output.split(b"\0") if value]


def main() -> None:
    failures: list[str] = []
    for path in tracked_files():
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden tracked artifact: {path.relative_to(ROOT)}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if path.name != ".env.example":
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"possible credential: {path.relative_to(ROOT)}")
        if "records" in path.parts and "selected_source_indices" in text:
            failures.append(f"heldout membership field: {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("Public release validation failed:\n- " + "\n- ".join(failures))
    print("Public release tree validation passed.")


if __name__ == "__main__":
    main()
