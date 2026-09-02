from __future__ import annotations

import json
from pathlib import Path


def test_leaderboard_is_sorted_and_separates_evaluation_regimes() -> None:
    root = Path(__file__).parents[1]
    board = json.loads((root / "records" / "leaderboard.json").read_text(encoding="utf-8"))
    assert board["schema"] == "nanoclassify.leaderboard.v1"
    assert set(board) >= {"development", "sealed_heldout"}
    for section in ("development", "sealed_heldout"):
        rows = board[section]
        assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
        assert [row["accuracy"] for row in rows] == sorted(
            (row["accuracy"] for row in rows), reverse=True
        )
    assert board["development"][0]["submission"] == "tinker-sft-cispo-v5-64r-50u"
