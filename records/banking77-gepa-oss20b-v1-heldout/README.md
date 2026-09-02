# Banking77 GEPA system-prompt run

This run used `synth-optimizers` 0.2.19 (`eaead59b03118fb3eaede1467e3ead9ab015b12b`) to optimize the Banking77 system prompt for Tinker's `openai/gpt-oss-20b` base model.

## Optimization protocol

- Candidate lever: `classification_system_prompt` only
- Proposer: Codex app server with ChatGPT authentication, `gpt-5.4-mini`, medium reasoning
- Task model: Tinker `openai/gpt-oss-20b`, temperature 0, low reasoning
- Search: 2 generations, 2 proposals per generation, 16-example minibatches
- Optimization data: 32 examples from the authoritative Banking77 training CSV
- Internal check: 16 disjoint examples from that training CSV
- Parallel rollout workers: 50
- Total optimizer usage: 128 task-model rollouts and 2 proposer calls; 340,497 tokens reported
- Provider-reported dollar cost: unavailable

The seed scored 25/32 (78.125%). All four proposed rewrites were rejected on their minibatches, so GEPA retained the seed. The retained prompt scored 8/16 (50%) on the internal check. There were no failed or degraded task-model rollouts.

## Locked heldout result

The retained prompt was then evaluated once on the canonical 400-example `banking77-hard-v5` lock with 50-way parallelism:

| Prompt | Accuracy | Macro F1 | Valid labels |
|---|---:|---:|---:|
| Previous base prompt | 35.25% (141/400) | 40.10% | 99.75% |
| GEPA run's retained seed | **39.25% (157/400)** | **43.91%** | 99.25% |

This is a +4.00 percentage-point accuracy change and +3.81-point macro-F1 change on the lock. Because GEPA did not improve on its own seed, this result supports the newly supplied seed prompt over the previous base prompt; it is not evidence that the search proposals improved the seed.

The exact prompt is in [`../../prompts/banking77-gepa-oss20b-v1.txt`](../../prompts/banking77-gepa-oss20b-v1.txt). `summary.json` contains dataset and prompt hashes, while `predictions.jsonl`, `per-class.json`, and `confusions.json` provide the complete evaluation evidence.
