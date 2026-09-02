#!/usr/bin/env python3
"""Evaluate the Tinker GPT-OSS-20B base model on a pinned Banking77 sample."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_SAMPLE_SEED = 20260901


def load_env_key(path: Path, name: str) -> None:
    """Load one named value from a dotenv file without logging or copying it."""

    if os.environ.get(name, "").strip():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().removeprefix("export ").strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ[name] = value
            return
    raise SystemExit(f"{name} is not set and was not found in {path}")


def normalized_label(text: str) -> str:
    return (text or "").strip().lower().replace("-", "_").replace(" ", "_")


def strict_label(text: str) -> str:
    return (text or "").strip()


def final_channel(text: str) -> str:
    """Extract GPT-OSS's native final channel from raw Harmony tokens."""

    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return text
    value = text.rsplit(marker, 1)[-1]
    for terminator in ("<|return|>", "<|end|>"):
        value = value.split(terminator, 1)[0]
    return value


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"held-out JSONL is empty: {path}")
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            raise SystemExit(f"row {index} does not contain a final assistant gold label")
    return rows


def select_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise SystemExit(f"sample size must be in 1..{len(rows)}, got {count}")
    indices = random.Random(seed).sample(range(len(rows)), count)
    return [dict(rows[index], source_index=index) for index in indices]


def select_indexed_rows(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    if not indices or len(indices) != len(set(indices)):
        raise SystemExit("explicit source indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(rows):
        raise SystemExit(f"explicit source indices must be in 0..{len(rows) - 1}")
    return [dict(rows[index], source_index=index) for index in indices]


def with_system_prompt(rows: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    """Return copied rows with only their system message replaced."""

    prompt = system_prompt.strip()
    if not prompt:
        raise SystemExit("system prompt override must not be empty")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = [dict(message) for message in row["messages"]]
        if not messages or messages[0].get("role") != "system":
            raise SystemExit(f"row {index} does not start with a system message")
        messages[0]["content"] = prompt
        result.append(dict(row, messages=messages))
    return result


def model_input(
    tinker: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    reasoning_effort: str,
) -> tuple[Any, int]:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    model_input_class = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
    try:
        value = model_input_class.from_ints(tokens=prompt_ids)
    except TypeError:
        value = model_input_class.from_ints(prompt_ids)
    return value, len(prompt_ids)


def sample_one(
    *,
    tinker: Any,
    sampling_client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    allowed_labels: set[str],
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    messages = row["messages"]
    gold = str(messages[-1].get("content") or "")
    prompt, prompt_tokens = model_input(tinker, tokenizer, messages[:-1], reasoning_effort)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    started = time.monotonic()
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=params,
    ).result()
    elapsed = time.monotonic() - started
    sequence = result.sequences[0]
    completion_ids = list(map(int, sequence.tokens))
    raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    prediction = final_channel(raw_completion)
    prediction_strict = strict_label(prediction)
    prediction_normalized = normalized_label(prediction)
    gold_normalized = normalized_label(gold)
    return {
        "source_index": row["source_index"],
        "task_id": (row.get("metadata") or {}).get("task_id", "banking77"),
        "world_ref": (row.get("metadata") or {}).get("world_ref"),
        "query": str(messages[-2].get("content") or ""),
        "gold": gold,
        "prediction": prediction,
        "raw_completion": raw_completion,
        "prediction_strict": prediction_strict,
        "prediction_normalized": prediction_normalized,
        "valid_label": prediction_normalized in allowed_labels,
        "strict_correct": prediction_strict == gold,
        "normalized_correct": prediction_normalized == gold_normalized,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(completion_ids),
        "latency_seconds": elapsed,
    }


def macro_f1(records: list[dict[str, Any]], labels: list[str]) -> tuple[float, dict[str, Any]]:
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        truth = normalized_label(label)
        true_positive = sum(
            row["gold_normalized"] == truth and row["prediction_normalized"] == truth
            for row in records
        )
        false_positive = sum(
            row["gold_normalized"] != truth and row["prediction_normalized"] == truth
            for row in records
        )
        false_negative = sum(
            row["gold_normalized"] == truth and row["prediction_normalized"] != truth
            for row in records
        )
        support = sum(row["gold_normalized"] == truth for row in records)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
        f1_values.append(f1)
    return statistics.fmean(f1_values), per_class


def confusion_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter(
        (row["gold"], row["prediction_strict"] if row["valid_label"] else "<invalid>")
        for row in records
        if not row["normalized_correct"]
    )
    return [
        {"gold": gold, "prediction": prediction, "count": count}
        for (gold, prediction), count in counts.most_common()
    ]


def sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_tinker_tls() -> None:
    """Include the macOS system certificate store in Tinker's HTTP transport."""

    try:
        import pyqwest
        import tinker._base_client as base_client
        from pyqwest.httpx import AsyncPyqwestTransport

        base_client._default_pyqwest_transport = lambda: AsyncPyqwestTransport(
            transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
        )
    except (ImportError, AttributeError, TypeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--source-indices-json", type=Path)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--model-path", help="Tinker sampler checkpoint; omit for base weights")
    parser.add_argument("--redact-membership", action="store_true",
                        help="omit selected source indices and private manifest digest")
    parser.add_argument("--parallelism", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    args = parser.parse_args()

    load_env_key(args.env_file, "TINKER_API_KEY")
    import tinker

    configure_tinker_tls()

    source_rows = jsonl_rows(args.heldout_jsonl)
    if args.source_indices_json:
        index_payload = json.loads(args.source_indices_json.read_text(encoding="utf-8"))
        indices = index_payload.get("selected_source_indices", index_payload)
        selected = select_indexed_rows(source_rows, [int(index) for index in indices])
        if len(selected) != args.sample_size:
            raise SystemExit(
                f"source index manifest contains {len(selected)} indices, expected {args.sample_size}"
            )
        selection_method = "explicit_source_indices"
    else:
        selected = select_rows(source_rows, count=args.sample_size, seed=args.sample_seed)
        selection_method = "random_without_replacement"
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        selected = with_system_prompt(selected, system_prompt)
    labels = sorted({str(row["messages"][-1]["content"]) for row in source_rows})
    allowed_normalized = {normalized_label(label) for label in labels}

    # The training client is used only to obtain Tinker's authoritative tokenizer
    # and chat template for this model. No training step is submitted.
    service = tinker.ServiceClient(user_metadata={"task": "nanoclassify_banking77_base_eval"})
    sampling_client = (
        service.create_sampling_client(model_path=args.model_path)
        if args.model_path
        else service.create_sampling_client(base_model=MODEL_ID)
    )
    tokenizer_client = service.create_lora_training_client(base_model=MODEL_ID, rank=8, seed=1)
    tokenizer = tokenizer_client.get_tokenizer()

    started_at = time.time()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = {
            executor.submit(
                sample_one,
                tinker=tinker,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                row=row,
                    allowed_labels=allowed_normalized,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
            ): row
            for row in selected
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - every provider failure belongs in the receipt
                errors.append({"source_index": row["source_index"], "error": type(exc).__name__})
            if completed % 50 == 0:
                print(json.dumps({"completed": completed, "total": len(futures), "errors": len(errors)}), flush=True)
    wall_seconds = time.time() - started_at
    records.sort(key=lambda row: int(row["source_index"]))
    for row in records:
        row["gold_normalized"] = normalized_label(str(row["gold"]))

    if len(records) != args.sample_size:
        raise SystemExit(f"only {len(records)}/{args.sample_size} samples completed; errors={len(errors)}")

    normalized_correct = sum(bool(row["normalized_correct"]) for row in records)
    strict_correct = sum(bool(row["strict_correct"]) for row in records)
    valid = sum(bool(row["valid_label"]) for row in records)
    macro, per_class = macro_f1(records, labels)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in records)
    completion_tokens = sum(int(row["completion_tokens"]) for row in records)
    summary = {
        "schema_version": "nanoclassify.baseline-receipt.v1",
        "task": "banking77",
        "track": "tinker",
        "model": args.model_path or MODEL_ID,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "selection_method": selection_method,
        "source_indices_json_sha256": sha256(args.source_indices_json) if args.source_indices_json else None,
        "system_prompt_file": args.system_prompt_file.name if args.system_prompt_file else None,
        "system_prompt_sha256": (
            f"sha256:{hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()}"
            if system_prompt is not None
            else None
        ),
        "parallelism": args.parallelism,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "accuracy": normalized_correct / args.sample_size,
        "strict_accuracy": strict_correct / args.sample_size,
        "macro_f1": macro,
        "valid_label_rate": valid / args.sample_size,
        "correct": normalized_correct,
        "strict_correct": strict_correct,
        "valid_labels": valid,
        "invalid_labels": args.sample_size - valid,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "wall_seconds": wall_seconds,
        "mean_request_latency_seconds": statistics.fmean(float(row["latency_seconds"]) for row in records),
        "heldout_jsonl_sha256": sha256(args.heldout_jsonl),
        "selected_source_indices": sorted(int(row["source_index"]) for row in records),
        "provider_reported_cost_usd": None,
        "note": (
            "Tinker sampler checkpoint evaluation."
            if args.model_path
            else "Base-model sampling only; tokenizer client created without optimizer steps."
        ),
    }
    if args.redact_membership:
        summary.pop("selected_source_indices", None)
        summary.pop("source_indices_json_sha256", None)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per-class.json", per_class)
    write_json(args.output_dir / "confusions.json", confusion_rows(records))
    write_json(args.output_dir / "errors.json", errors)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
