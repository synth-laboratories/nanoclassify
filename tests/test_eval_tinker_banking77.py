from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_tinker_banking77.py"
SPEC = importlib.util.spec_from_file_location("eval_tinker_banking77", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(index: int, label: str = "card_arrival") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": f"query {index}"},
            {"role": "assistant", "content": label},
        ]
    }


def test_selection_is_unique_and_deterministic() -> None:
    rows = [row(index) for index in range(1000)]
    first = MODULE.select_rows(rows, count=400, seed=20260901)
    second = MODULE.select_rows(rows, count=400, seed=20260901)
    first_ids = [item["source_index"] for item in first]
    second_ids = [item["source_index"] for item in second]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)) == 400


def test_explicit_index_selection_preserves_order() -> None:
    rows = [row(index) for index in range(10)]
    selected = MODULE.select_indexed_rows(rows, [7, 2, 9])
    assert [item["source_index"] for item in selected] == [7, 2, 9]


def test_system_prompt_override_does_not_mutate_source_rows() -> None:
    rows = [row(0), row(1)]
    selected = MODULE.with_system_prompt(rows, "  optimized prompt  ")
    assert [item["messages"][0]["content"] for item in selected] == [
        "optimized prompt",
        "optimized prompt",
    ]
    assert rows[0]["messages"][0]["content"] == "classify"


def test_normalization_matches_container_semantics() -> None:
    assert MODULE.normalized_label(" Card Arrival ") == "card_arrival"
    assert MODULE.normalized_label("card-arrival") == "card_arrival"


def test_gpt_oss_final_channel_is_scored() -> None:
    raw = (
        "<|channel|>analysis<|message|>reasoning<|end|>"
        "<|start|>assistant<|channel|>final<|message|>terminate_account<|return|>"
    )
    assert MODULE.final_channel(raw) == "terminate_account"


def test_macro_f1_reports_all_labels() -> None:
    records = [
        {"gold_normalized": "a", "prediction_normalized": "a"},
        {"gold_normalized": "b", "prediction_normalized": "a"},
    ]
    score, per_class = MODULE.macro_f1(records, ["a", "b"])
    assert round(score, 6) == round((2 / 3 + 0) / 2, 6)
    assert per_class["b"]["support"] == 1
