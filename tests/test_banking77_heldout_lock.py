from __future__ import annotations

from pathlib import Path


def test_public_tree_does_not_ship_heldout_membership() -> None:
    root = Path(__file__).parents[1]
    assert not (root / "tasks" / "banking77" / "heldout.lock.json").exists()
    assert not list((root / "manifests").glob("banking77-hard-*.json"))
