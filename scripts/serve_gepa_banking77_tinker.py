#!/usr/bin/env python3
"""Serve the optimizers-package Banking77 GEPA contract with a Tinker task model."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

MODEL_ID = "openai/gpt-oss-20b"
LEGACY_ROOT = Path("/Users/joshuapurtell/GitHub/evals/temp/banking77-gepa")
DEFAULT_TRAIN_CSV = Path("/tmp/banking77-cache/banking77-train.csv")
ROOT = Path(__file__).parents[1]


def load_env_key(path: Path, name: str) -> None:
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


def configure_tinker_tls() -> None:
    try:
        import pyqwest
        import tinker._base_client as base_client
        from pyqwest.httpx import AsyncPyqwestTransport

        base_client._default_pyqwest_transport = lambda: AsyncPyqwestTransport(
            transport=pyqwest.HTTPTransport(tls_include_system_certs=True)
        )
    except (ImportError, AttributeError, TypeError):
        pass


def final_channel(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return text
    value = text.rsplit(marker, 1)[-1]
    for terminator in ("<|return|>", "<|end|>"):
        value = value.split(terminator, 1)[0]
    return value


def load_training_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {"id": f"banking77-train-{index}", "query": row["text"], "expected": row["category"]}
            for index, row in enumerate(csv.DictReader(handle))
            if row.get("text") and row.get("category")
        ]


def main() -> None:
    env_file = Path(os.environ.get("NANOCLASSIFY_ENV_FILE", ROOT / ".env.local"))
    train_csv = Path(os.environ.get("NANOCLASSIFY_BANKING77_TRAIN_CSV", DEFAULT_TRAIN_CSV))
    load_env_key(env_file, "TINKER_API_KEY")
    configure_tinker_tls()

    import tinker

    sys.path.insert(0, str(LEGACY_ROOT))
    import serve as facade

    rows = load_training_rows(train_csv)
    labels = tuple(sorted({row["expected"] for row in rows}))
    if len(rows) != 10003 or len(labels) != 77:
        raise SystemExit(f"expected 10003 training rows and 77 labels, got {len(rows)} and {len(labels)}")
    facade.DATASET_PATH = train_csv
    facade.DATASET_SOURCE = "PolyAI-LDN/task-specific-datasets:banking_data/train.csv"
    facade.DATASET_CONFIG = "train"
    facade.DATASET_REVISION = "nanoclassify:banking77-train-authoritative-v1"
    facade.ROWS = rows
    facade.LABELS = labels
    facade.LABEL_SET = frozenset(labels)

    service = tinker.ServiceClient(user_metadata={"task": "nanoclassify_banking77_gepa"})
    sampling_client = service.create_sampling_client(base_model=MODEL_ID)
    tokenizer_client = service.create_lora_training_client(base_model=MODEL_ID, rank=8, seed=1)
    tokenizer = tokenizer_client.get_tokenizer()
    delivery_path = ROOT / "artifacts" / "gepa" / "prompt-delivery.jsonl"
    delivery_path.parent.mkdir(parents=True, exist_ok=True)
    delivery_lock = threading.Lock()

    def tinker_completion(
        route: str,
        model: str,
        prompt: str,
        query: str,
        timeout: float,
        reasoning_effort: str = "low",
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        del route, model, timeout
        effective_max_tokens = max(int(max_tokens), 1024)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt}, {"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        prompt_ids = list(map(int, tokenizer(rendered, add_special_tokens=False)["input_ids"]))
        model_input_cls = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
        try:
            model_input = model_input_cls.from_ints(tokens=prompt_ids)
        except TypeError:
            model_input = model_input_cls.from_ints(prompt_ids)
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=effective_max_tokens, temperature=0.0),
        ).result()
        completion_ids = list(map(int, result.sequences[0].tokens))
        raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
        with delivery_lock, delivery_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "reasoning_effort": reasoning_effort,
                        "max_tokens": effective_max_tokens,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return final_channel(raw).strip(), {
            "input_tokens": len(prompt_ids),
            "output_tokens": len(completion_ids),
            "total_tokens": len(prompt_ids) + len(completion_ids),
            "cost_usd": None,
        }

    facade._chat_completion = tinker_completion
    port = int(os.environ.get("PORT", "18877"))
    state_dir = Path(os.environ.get("NANOCLASSIFY_GEPA_STATE_DIR", ROOT / "artifacts/gepa/runtime"))
    server = facade.create_server(host="127.0.0.1", port=port, state_dir=state_dir)
    print(
        json.dumps(
            {
                "endpoint": f"http://127.0.0.1:{port}",
                "model": MODEL_ID,
                "rows": len(rows),
                "labels": len(labels),
                "dataset": facade.dataset_manifest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
