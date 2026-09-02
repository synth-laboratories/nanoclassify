#!/usr/bin/env python3
"""Retain failures from one Banking77 round and replace its passing examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--heldout-size", required=True, type=int)
    parser.add_argument("--replacement-seed", required=True, type=int)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original = sorted(int(row["source_index"]) for row in rows)
    retained = sorted(int(row["source_index"]) for row in rows if not row["normalized_correct"])
    removed = sorted(set(original) - set(retained))
    previously_seen = set(original)
    parent_manifest_sha256 = None
    if args.prior_manifest:
        prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
        previously_seen.update(
            int(index)
            for index in prior.get(
                "seen_source_indices",
                prior.get("parent_source_indices", []) + prior.get("replacement_source_indices", []),
            )
        )
        parent_manifest_sha256 = sha256(args.prior_manifest)
    candidate_pool = sorted(set(range(args.heldout_size)) - previously_seen)
    if len(candidate_pool) < len(removed):
        raise SystemExit(
            f"only {len(candidate_pool)} never-seen examples remain for {len(removed)} replacements"
        )
    replacements = sorted(random.Random(args.replacement_seed).sample(candidate_pool, len(removed)))
    selected = sorted(retained + replacements)

    payload = {
        "schema_version": "nanoclassify.hardening-manifest.v1",
        "strategy": "retain_failures_replace_passes_without_prior_round_reuse",
        "parent_predictions_sha256": sha256(args.predictions_jsonl),
        "parent_manifest_sha256": parent_manifest_sha256,
        "replacement_seed": args.replacement_seed,
        "heldout_size": args.heldout_size,
        "parent_source_indices": original,
        "retained_failure_source_indices": retained,
        "removed_pass_source_indices": removed,
        "replacement_source_indices": replacements,
        "selected_source_indices": selected,
        "seen_source_indices": sorted(previously_seen | set(replacements)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "retained_failures": len(retained),
                "removed_passes": len(removed),
                "replacements": len(replacements),
                "selected": len(selected),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
