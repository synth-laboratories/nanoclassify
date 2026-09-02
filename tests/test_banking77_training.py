from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nanoclassify.banking77 import digest_rows, stratified_split, system_prompt


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    rows = [
        {"text": f"{label}-{i}", "category": label}
        for label in ("alpha", "beta", "gamma")
        for i in range(6)
    ]
    train, dev = stratified_split(rows, seed=7, dev_per_class=2)
    again = stratified_split(rows, seed=7, dev_per_class=2)
    assert (train, dev) == again
    assert len(dev) == 6
    assert {row["text"] for row in train}.isdisjoint(row["text"] for row in dev)
    assert digest_rows(train).startswith("sha256:")


def test_prompt_contains_every_label() -> None:
    prompt = system_prompt(["a", "b"])
    assert "a, b" in prompt
    assert "exactly one label" in prompt
