# Banking77 locked challenge heldout

The canonical 400-example Banking77 challenge heldout is maintained privately.
Its membership and gold answers are never shipped in the public repository.

## Immutable contract

- Selected examples: `400` unique source indices
- Held-out population: `3,080`
- Banking77 container dataset digest: `sha256:9b52cfe49372fa6e6ce41654de86d641ac7fff56a89d1483b281f3e9732ff4dc`
- Exported held-out JSONL digest: `sha256:195988a7db018fb6bdee632f96f1e676047d7ebebf788a16afb1027b9e8d6fb4`
- Mining baseline: `openai/gpt-oss-20b` through Tinker
- Mining baseline accuracy: `35.25%` (`141/400`)
- Mining baseline macro-F1: `40.10%`

Maintainers verify membership against the private manifest before every official run.
A future heldout definition requires a new version and an explicit migration.

This set was deliberately mined against the GPT-OSS-20B base model over six rounds.
It is a fixed challenge benchmark, not an IID estimate of Banking77 generalization.
The 1,461 never-exposed examples remain outside this challenge set.

## Evaluation

Maintainers pass the private canonical lock directly to the evaluator:

```bash
python scripts/eval_tinker_banking77.py \
  --heldout-jsonl /path/to/authoritative/heldout.jsonl \
  --source-indices-json /path/to/private/heldout.lock.json \
  --sample-size 400 \
  --sample-seed 20260906 \
  --parallelism 50 \
  --max-tokens 1024 \
  --reasoning-effort low \
  --env-file .env.local \
  --output-dir records/<candidate-name>
```
