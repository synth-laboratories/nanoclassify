# Banking77 large-seed GEPA v5

This experiment used `synth-optimizers` 0.2.19 with Tinker's frozen `openai/gpt-oss-20b` base model and a `gpt-5.4` high-reasoning proposer. Only `classification_system_prompt` was mutable.

## Search and selection

- Optimization pool: 1,200 examples from the authoritative Banking77 training partition
- Matched reflection/minibatch: 400 examples
- Internal validation: 400 disjoint examples
- Parallel rollout workers: 50
- Canonical lock: the frozen 400-example `banking77-hard-v5` manifest

The selected v5 proposal scored 77.25% on its 400-example search minibatch versus 69.50% for the matched parent (+7.75 points). In the clean confirmation runs it scored 75.33% on the 1,200-example optimization pool versus 75.75% for the seed, and 74.25% on the disjoint internal set versus 73.75% for the seed. The large minibatch gain therefore did not reproduce on the full optimization pool, but the proposal did transfer positively to the disjoint internal set (+0.50 points).

It was then evaluated once on the canonical locked hard set:

| Prompt | Accuracy | Macro F1 | Valid-label rate |
|---|---:|---:|---:|
| Original locked baseline | 35.25% (141/400) | 40.10% | 99.75% |
| Earlier GEPA seed | 39.25% (157/400) | 43.91% | 99.25% |
| **Large-seed GEPA v5** | **44.75% (179/400)** | **48.19%** | 97.25% |

The v5 prompt improves locked accuracy by 5.50 points over the earlier GEPA result and 9.50 points over the original baseline. Macro F1 improves by 4.28 and 8.08 points respectively. The tradeoff is 11 invalid-label outputs, versus 3 for the earlier GEPA prompt.

The lock has now been consulted for two GEPA prompt versions, so subsequent prompt selection should use a newly defined validation protocol rather than repeatedly tuning against this lock.

The exact prompt is in [`../../prompts/banking77-gepa-oss20b-large-v5-frontier.txt`](../../prompts/banking77-gepa-oss20b-large-v5-frontier.txt). `summary.json`, `predictions.jsonl`, `per-class.json`, and `confusions.json` contain the complete locked evaluation evidence.
