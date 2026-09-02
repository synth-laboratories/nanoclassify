"""Deterministic Banking77 data and prompt helpers."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or set(rows[0]) != {"text", "category"}:
        raise ValueError("expected Banking77 CSV columns: text, category")
    return rows


def labels(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["category"] for row in rows})


def system_prompt(label_names: list[str]) -> str:
    return (
        "Classify the customer banking message. Return exactly one label from this list, "
        "with no explanation or punctuation:\n" + ", ".join(label_names)
    )


def stratified_split(
    rows: list[dict[str, str]], *, seed: int, dev_per_class: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    dev: list[dict[str, str]] = []
    for category in sorted(grouped):
        bucket = grouped[category][:]
        rng.shuffle(bucket)
        dev.extend(bucket[:dev_per_class])
        train.extend(bucket[dev_per_class:])
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def digest_rows(rows: list[dict[str, str]]) -> str:
    payload = "".join(f'{row["category"]}\t{row["text"]}\n' for row in rows)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
