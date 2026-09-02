# Banking77 · Tinker GPT-OSS-20B base baseline

This receipt evaluates the untrained `openai/gpt-oss-20b` base sampling client on
400 unique examples selected without replacement from the authoritative Banking77
held-out split.

- Sample seed: `20260901`
- Held-out population: `3,080`
- Sample size: `400`
- Parallelism: `50`
- Temperature: `0.0`
- Reasoning effort: `low`
- Maximum completion: `1,024` tokens
- Banking77 container dataset digest: `sha256:9b52cfe49372fa6e6ce41654de86d641ac7fff56a89d1483b281f3e9732ff4dc`
- Exported held-out JSONL digest: `sha256:195988a7db018fb6bdee632f96f1e676047d7ebebf788a16afb1027b9e8d6fb4`

The run made no optimizer or training steps. A Tinker training client is instantiated
only to obtain the model's authoritative tokenizer. One output returned the invented
near-synonym `top_up_by_card`, which is outside the canonical Banking77 vocabulary
and is retained as an invalid prediction. No output exhausted the 1,024-token ceiling,
and there were no provider request errors.

## Reproduce

Export the held-out JSONL through the pinned Banking77 container, then run:

```bash
python scripts/eval_tinker_banking77.py \
  --heldout-jsonl /path/to/heldout.jsonl \
  --output-dir records/banking77-tinker-gpt-oss-20b-base-400-seed-20260901 \
  --env-file .env.local \
  --sample-size 400 \
  --sample-seed 20260901 \
  --parallelism 50 \
  --max-tokens 1024 \
  --reasoning-effort low
```

Only `TINKER_API_KEY` is loaded from the project-local environment file. Provider
cost was not reported by the Tinker sampling response; token counts are recorded in
`summary.json`.
