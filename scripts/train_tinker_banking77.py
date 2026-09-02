#!/usr/bin/env python3
"""Train reproducible GPT-OSS-20B Banking77 SFT and optional CISPO adapters."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanoclassify.banking77 import digest_rows, labels, load_csv, stratified_split, system_prompt

MODEL_ID = "openai/gpt-oss-20b"


def final_answer(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    for terminator in ("<|return|>", "<|end|>"):
        text = text.split(terminator, 1)[0]
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def load_key(path: Path) -> None:
    if os.environ.get("TINKER_API_KEY", "").strip():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = raw.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit("TINKER_API_KEY is unavailable")


def tls() -> None:
    try:
        import pyqwest
        import tinker._base_client as base_client
        from pyqwest.httpx import AsyncPyqwestTransport
        base_client._default_pyqwest_transport = lambda: AsyncPyqwestTransport(
            transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
        )
    except (ImportError, AttributeError, TypeError):
        pass


def token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def sft_datum(row: dict[str, str], prompt: str, tokenizer: Any, tinker: Any) -> tuple[Any, int]:
    prefix = token_ids(tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": row["text"]}],
        tokenize=True, add_generation_prompt=True,
    ))
    full = token_ids(tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": row["text"]},
         {"role": "assistant", "content": row["category"]}],
        tokenize=True, add_generation_prompt=False,
    ))
    if full[:len(prefix)] != prefix:
        raise ValueError("chat template is not prefix-stable")
    weights = [0.0] * len(prefix) + [1.0] * (len(full) - len(prefix))
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=full[1:], dtype="int64", shape=[len(full)-1]),
            "weights": tinker.TensorData(data=weights[1:], dtype="float32", shape=[len(full)-1]),
        },
    )
    return datum, len(full) - 1


def rollout(client: Any, tokenizer: Any, tinker: Any, row: dict[str, str], prompt_text: str,
            seed: int, temperature: float) -> dict[str, Any]:
    prefix = token_ids(tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt_text}, {"role": "user", "content": row["text"]}],
        tokenize=True, add_generation_prompt=True,
    ))
    result = client.sample(
        prompt=tinker.ModelInput.from_ints(prefix), num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=24, temperature=temperature, seed=seed),
    ).result().sequences[0]
    completion = [int(x) for x in result.tokens]
    answer = final_answer(tokenizer.decode(completion, skip_special_tokens=False))
    return {"tokens": prefix + completion, "prompt_len": len(prefix),
            "logprobs": [float(x) for x in result.logprobs],
            "reward": float(answer == row["category"]), "answer": answer}


def cispo_datum(sample: dict[str, Any], advantage: float, tinker: Any) -> Any:
    ids = sample["tokens"]
    prompt_len = sample["prompt_len"]
    shifted_train = [i >= prompt_len for i in range(1, len(ids))]
    trained = sum(shifted_train)
    completion_logprobs = iter(sample["logprobs"])
    logprobs, advantages = [], []
    for enabled in shifted_train:
        logprobs.append(next(completion_logprobs) if enabled else 0.0)
        advantages.append(advantage / trained if enabled and trained else 0.0)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=ids[1:], dtype="int64", shape=[len(ids)-1]),
            "logprobs": tinker.TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)]),
            "advantages": tinker.TensorData(data=advantages, dtype="float32", shape=[len(advantages)]),
        },
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rollout_groups(path: Path, groups: list[list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group_id, group in enumerate(groups):
            handle.write(json.dumps({"schema": "nanoclassify.cispo-rollout-group.v1",
                                     "group_id": group_id, "rollouts": group}, sort_keys=True) + "\n")


def read_rollout_groups(path: Path) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "nanoclassify.cispo-rollout-group.v1":
            raise SystemExit("unexpected rollout-group schema")
        groups.append(list(row["rollouts"]))
    if not groups:
        raise SystemExit("rollout-group file is empty")
    return groups


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--env-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--dev-per-class", type=int, default=10)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--sft-updates", type=int, default=120)
    p.add_argument("--sft-batch-size", type=int, default=32)
    p.add_argument("--sft-learning-rate", type=float, default=2e-5)
    p.add_argument("--parent-sft-json", type=Path,
                   help="Continue CISPO from an existing sft.json instead of retraining")
    p.add_argument("--cispo-updates", type=int, default=12)
    p.add_argument("--cispo-prompts-per-update", type=int, default=16)
    p.add_argument("--rollouts-per-prompt", type=int, default=4)
    p.add_argument("--cispo-upfront-prompts", type=int, default=0,
                   help="collect this many prompt groups once from the parent sampler")
    p.add_argument("--save-eligible-rollouts", type=Path,
                   help="persist qualifying tokenized rollout groups as private JSONL")
    p.add_argument("--eligible-rollouts-jsonl", type=Path,
                   help="skip sampling and train CISPO from a saved qualifying JSONL")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--qualify-seeds-output", type=Path,
                   help="qualification-only: persist immutable qualifying seed IDs")
    p.add_argument("--qualifying-seeds-json", type=Path,
                   help="CISPO: load immutable qualifying seed IDs and sample training rollouts")
    p.add_argument("--cispo-groups-per-update", type=int, default=16)
    p.add_argument("--cispo-target-updates", type=int, default=0,
                   help="run exactly this many updates, replaying saved groups as needed")
    p.add_argument("--cispo-min-correct", type=int, default=1)
    p.add_argument("--cispo-max-correct", type=int, default=5)
    p.add_argument("--cispo-learning-rate", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--parallelism", type=int, default=50)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory already exists")
    args.output_dir.mkdir(parents=True)
    load_key(args.env_file)
    tls()
    import tinker

    all_rows = [dict(row, _source_index=index)
                for index, row in enumerate(load_csv(args.train_csv))]
    train, dev = stratified_split(all_rows, seed=args.seed, dev_per_class=args.dev_per_class)
    prompt_text = system_prompt(labels(all_rows))
    (args.output_dir / "system-prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")
    manifest = {"schema": "nanoclassify.training-split.v1", "seed": args.seed,
                "source_digest": digest_rows(all_rows), "train_digest": digest_rows(train),
                "dev_digest": digest_rows(dev), "train_count": len(train), "dev_count": len(dev)}
    write_json(args.output_dir / "split-manifest.json", manifest)

    service = tinker.ServiceClient(user_metadata={"project": "nanoclassify", "task": "banking77"})
    tokenizer_client = service.create_lora_training_client(base_model=MODEL_ID, rank=args.rank, seed=args.seed)
    tokenizer = tokenizer_client.get_tokenizer()
    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    started = time.time()
    if args.parent_sft_json:
        parent = json.loads(args.parent_sft_json.read_text(encoding="utf-8"))
        sft_state = str(parent["state_checkpoint"])
        sft_sampler = str(parent["sampler_checkpoint"])
        write_json(args.output_dir / "sft-parent.json", {
            "schema": "nanoclassify.tinker-sft-parent.v1", "path": str(args.parent_sft_json),
            "state_checkpoint": sft_state, "sampler_checkpoint": sft_sampler})
    else:
        trainer = tokenizer_client
        for update in range(1, args.sft_updates + 1):
            batch = rng.sample(train, args.sft_batch_size)
            prepared = [sft_datum(row, prompt_text, tokenizer, tinker) for row in batch]
            result = trainer.forward_backward([x[0] for x in prepared], loss_fn="cross_entropy").result()
            trainer.optim_step(tinker.AdamParams(learning_rate=args.sft_learning_rate)).result()
            history.append({"update": update, "tokens": sum(x[1] for x in prepared), "metrics": result.metrics})
            if update % 10 == 0:
                print(json.dumps({"phase": "sft", "update": update, "of": args.sft_updates}), flush=True)
        sft_state = str(trainer.save_state("nanoclassify-banking77-sft-state", ttl_seconds=30*86400).result().path)
        sft_sampler = str(trainer.save_weights_for_sampler(
            "nanoclassify-banking77-sft-sampler", ttl_seconds=30*86400).result().path)
        write_json(args.output_dir / "sft.json", {"schema": "nanoclassify.tinker-sft.v1",
            "base_model": MODEL_ID, "state_checkpoint": sft_state, "sampler_checkpoint": sft_sampler,
            "updates": args.sft_updates, "batch_size": args.sft_batch_size,
            "learning_rate": args.sft_learning_rate, "history": history,
            "wall_seconds": time.time() - started})

    if args.cispo_updates <= 0:
        return
    trainer = service.create_training_client_from_state(
        sft_state, user_metadata={"objective": "cispo", "optimizer_reset": "true"})
    current_sampler = sft_sampler
    cispo_history: list[dict[str, Any]] = []
    if args.qualify_seeds_output:
        if not args.cispo_upfront_prompts:
            raise SystemExit("qualification requires --cispo-upfront-prompts")
        prompts = rng.sample(train, args.cispo_upfront_prompts)
        sampler = service.create_sampling_client(model_path=sft_sampler)
        grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(prompts))}
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            for pi, row in enumerate(prompts):
                for ri in range(args.rollouts_per_prompt):
                    sample_seed = args.seed + pi * args.rollouts_per_prompt + ri
                    jobs.append((pi, pool.submit(
                        rollout, sampler, tokenizer, tinker, row, prompt_text,
                        sample_seed, args.temperature,
                    )))
            for completed, (pi, future) in enumerate(jobs, start=1):
                grouped[pi].append(future.result())
                if completed % 500 == 0:
                    print(json.dumps({"phase": "qualify", "completed": completed,
                                      "of": len(jobs)}), flush=True)
        qualified: list[dict[str, int]] = []
        histogram = {str(value): 0 for value in range(args.rollouts_per_prompt + 1)}
        for pi, group in grouped.items():
            correct = int(sum(item["reward"] for item in group))
            histogram[str(correct)] += 1
            if args.cispo_min_correct <= correct <= args.cispo_max_correct:
                qualified.append({"source_index": int(prompts[pi]["_source_index"]),
                                  "qualification_correct": correct})
        private_manifest = {
            "schema": "nanoclassify.qualifying-seeds.v1", "task": "banking77",
            "source_digest": digest_rows(all_rows), "split_seed": args.seed,
            "qualification_prompt_groups": len(prompts),
            "qualification_rollouts_per_seed": args.rollouts_per_prompt,
            "qualification_correctness_range": [args.cispo_min_correct, args.cispo_max_correct],
            "qualifying_seeds": qualified,
        }
        args.qualify_seeds_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.qualify_seeds_output, private_manifest)
        public_receipt = {
            "schema": "nanoclassify.seed-qualification-receipt.v1",
            "qualification_prompt_groups": len(prompts),
            "qualification_rollouts_per_seed": args.rollouts_per_prompt,
            "qualification_correctness_range": [args.cispo_min_correct, args.cispo_max_correct],
            "correctness_histogram": histogram, "qualifying_seed_count": len(qualified),
            "private_manifest_sha256": file_sha256(args.qualify_seeds_output),
        }
        write_json(args.output_dir / "qualification.json", public_receipt)
        print(json.dumps({"phase": "qualified", **public_receipt}), flush=True)
        return
    if args.qualifying_seeds_json:
        seed_manifest = json.loads(args.qualifying_seeds_json.read_text(encoding="utf-8"))
        if seed_manifest.get("schema") != "nanoclassify.qualifying-seeds.v1":
            raise SystemExit("unexpected qualifying-seed manifest schema")
        if seed_manifest.get("source_digest") != digest_rows(all_rows):
            raise SystemExit("qualifying-seed manifest source digest mismatch")
        by_index = {int(row["_source_index"]): row for row in train}
        prompts = [by_index[int(item["source_index"])]
                   for item in seed_manifest["qualifying_seeds"]]
        sampler = service.create_sampling_client(model_path=sft_sampler)
        grouped = {i: [] for i in range(len(prompts))}
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            for pi, row in enumerate(prompts):
                for ri in range(args.rollouts_per_prompt):
                    sample_seed = args.seed + 10_000_000 + pi * args.rollouts_per_prompt + ri
                    jobs.append((pi, pool.submit(
                        rollout, sampler, tokenizer, tinker, row, prompt_text,
                        sample_seed, args.temperature,
                    )))
            for completed, (pi, future) in enumerate(jobs, start=1):
                grouped[pi].append(future.result())
                if completed % 500 == 0:
                    print(json.dumps({"phase": "cispo-rollouts", "completed": completed,
                                      "of": len(jobs)}), flush=True)
        eligible = list(grouped.values())
        collection = {
            "schema": "nanoclassify.cispo-qualified-seed-rollouts.v1",
            "qualifying_manifest_sha256": file_sha256(args.qualifying_seeds_json),
            "seed_groups": len(eligible), "rollouts_per_prompt": args.rollouts_per_prompt,
            "total_rollouts": len(eligible) * args.rollouts_per_prompt,
        }
        write_json(args.output_dir / "upfront-collection.json", collection)
        print(json.dumps({"phase": "cispo-rollouts-complete", **collection}), flush=True)
    elif args.eligible_rollouts_jsonl:
        eligible = read_rollout_groups(args.eligible_rollouts_jsonl)
        if any(len(group) != args.rollouts_per_prompt for group in eligible):
            raise SystemExit("saved rollout group size does not match --rollouts-per-prompt")
        collection = {
            "schema": "nanoclassify.cispo-saved-collection.v1",
            "source": str(args.eligible_rollouts_jsonl),
            "eligible_groups": len(eligible),
            "eligible_rollouts": sum(len(group) for group in eligible),
            "rollouts_per_prompt": args.rollouts_per_prompt,
        }
        write_json(args.output_dir / "upfront-collection.json", collection)
        print(json.dumps({"phase": "load", **collection}), flush=True)
    elif args.cispo_upfront_prompts:
        if args.cispo_upfront_prompts > len(train):
            raise SystemExit("upfront prompt count exceeds the training split")
        if not 0 <= args.cispo_min_correct <= args.cispo_max_correct <= args.rollouts_per_prompt:
            raise SystemExit("correctness range must fit within the rollout count")
        prompts = rng.sample(train, args.cispo_upfront_prompts)
        sampler = service.create_sampling_client(model_path=sft_sampler)
        grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(prompts))}
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            for pi, row in enumerate(prompts):
                for ri in range(args.rollouts_per_prompt):
                    sample_seed = args.seed + pi * args.rollouts_per_prompt + ri
                    jobs.append((pi, pool.submit(
                        rollout, sampler, tokenizer, tinker, row, prompt_text,
                        sample_seed, args.temperature,
                    )))
            for completed, (pi, future) in enumerate(jobs, start=1):
                grouped[pi].append(future.result())
                if completed % 500 == 0:
                    print(json.dumps({"phase": "collect", "completed": completed,
                                      "of": len(jobs)}), flush=True)
        correctness_histogram = {str(value): 0 for value in range(args.rollouts_per_prompt + 1)}
        eligible: list[list[dict[str, Any]]] = []
        for group in grouped.values():
            correct = int(sum(item["reward"] for item in group))
            correctness_histogram[str(correct)] += 1
            if args.cispo_min_correct <= correct <= args.cispo_max_correct:
                eligible.append(group)
        rng.shuffle(eligible)
        collection = {
            "schema": "nanoclassify.cispo-upfront-collection.v1",
            "parent_sampler": sft_sampler,
            "prompt_groups": len(prompts),
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "total_rollouts": len(prompts) * args.rollouts_per_prompt,
            "correctness_histogram": correctness_histogram,
            "eligible_correctness_range": [args.cispo_min_correct, args.cispo_max_correct],
            "eligible_groups": len(eligible),
            "eligible_rollouts": len(eligible) * args.rollouts_per_prompt,
        }
        write_json(args.output_dir / "upfront-collection.json", collection)
        if args.save_eligible_rollouts:
            write_rollout_groups(args.save_eligible_rollouts, eligible)
            collection["saved_rollout_groups"] = str(args.save_eligible_rollouts)
            write_json(args.output_dir / "upfront-collection.json", collection)
        print(json.dumps({"phase": "filter", **collection}), flush=True)
        if args.collect_only:
            return
    else:
        eligible = []
    if args.cispo_upfront_prompts or args.eligible_rollouts_jsonl or args.qualifying_seeds_json:
        batches: list[list[list[dict[str, Any]]]] = []
        if args.cispo_target_updates:
            pool = eligible[:]
            while len(batches) < args.cispo_target_updates:
                rng.shuffle(pool)
                for offset in range(0, len(pool), args.cispo_groups_per_update):
                    batches.append(pool[offset:offset + args.cispo_groups_per_update])
                    if len(batches) == args.cispo_target_updates:
                        break
        else:
            batches = [eligible[offset:offset + args.cispo_groups_per_update]
                       for offset in range(0, len(eligible), args.cispo_groups_per_update)]
        groups_trained = 0
        for update, batch_groups in enumerate(batches, start=1):
            data: list[Any] = []
            correct_counts: list[int] = []
            for group in batch_groups:
                mean = sum(item["reward"] for item in group) / len(group)
                correct_counts.append(int(sum(item["reward"] for item in group)))
                data.extend(cispo_datum(item, item["reward"] - mean, tinker) for item in group)
            result = trainer.forward_backward(
                data, loss_fn="cispo",
                loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 2.0},
            ).result()
            trainer.optim_step(tinker.AdamParams(learning_rate=args.cispo_learning_rate)).result()
            groups_trained += len(batch_groups)
            cispo_history.append({
                "update": update, "group_count": len(batch_groups),
                "correct_counts": correct_counts, "trajectories_trained": len(data),
                "metrics": result.metrics,
            })
            print(json.dumps({"phase": "cispo", "update": update,
                              "groups_trained": groups_trained,
                              "unique_groups": len(eligible),
                              "target_updates": len(batches)}), flush=True)
        current_sampler = str(trainer.save_weights_for_sampler(
            "nanoclassify-banking77-cispo-upfront", ttl_seconds=30*86400).result().path)
        final_state = str(trainer.save_state(
            "nanoclassify-banking77-cispo-upfront-state", ttl_seconds=30*86400).result().path)
        write_json(args.output_dir / "cispo.json", {
            "schema": "nanoclassify.tinker-cispo-upfront.v1",
            "parent_sampler": sft_sampler, "state_checkpoint": final_state,
            "sampler_checkpoint": current_sampler, "collection": collection,
            "learning_rate": args.cispo_learning_rate, "groups_per_update": args.cispo_groups_per_update,
            "target_updates": args.cispo_target_updates or len(batches),
            "history": cispo_history,
        })
        return
    for update in range(1, args.cispo_updates + 1):
        sampler = service.create_sampling_client(model_path=current_sampler)
        prompts = rng.sample(train, args.cispo_prompts_per_update)
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            for pi, row in enumerate(prompts):
                for ri in range(args.rollouts_per_prompt):
                    jobs.append((pi, pool.submit(rollout, sampler, tokenizer, tinker, row, prompt_text,
                                                 args.seed + update*10000 + pi*100 + ri, args.temperature)))
            grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(prompts))}
            for pi, future in jobs:
                grouped[pi].append(future.result())
        data, rewards, mixed = [], [], 0
        for group in grouped.values():
            mean = sum(x["reward"] for x in group) / len(group)
            rewards.extend(x["reward"] for x in group)
            if 0.0 < mean < 1.0:
                mixed += 1
                data.extend(cispo_datum(x, x["reward"] - mean, tinker) for x in group)
        if data:
            result = trainer.forward_backward(data, loss_fn="cispo",
                loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 2.0}).result()
            trainer.optim_step(tinker.AdamParams(learning_rate=args.cispo_learning_rate)).result()
            metrics = result.metrics
        else:
            metrics = {"skipped": 1.0}
        if update == args.cispo_updates or update % 4 == 0:
            current_sampler = str(trainer.save_weights_for_sampler(
                f"nanoclassify-banking77-cispo-u{update:03d}", ttl_seconds=30*86400).result().path)
        cispo_history.append({"update": update, "mean_reward": sum(rewards)/len(rewards),
                              "mixed_groups": mixed, "trajectories_trained": len(data), "metrics": metrics,
                              "sampler_checkpoint": current_sampler})
        print(json.dumps({"phase": "cispo", **cispo_history[-1]}), flush=True)
    final_state = str(trainer.save_state("nanoclassify-banking77-cispo-state", ttl_seconds=30*86400).result().path)
    write_json(args.output_dir / "cispo.json", {"schema": "nanoclassify.tinker-cispo.v1",
        "parent_sampler": sft_sampler, "state_checkpoint": final_state,
        "sampler_checkpoint": current_sampler, "history": cispo_history})


if __name__ == "__main__":
    main()
