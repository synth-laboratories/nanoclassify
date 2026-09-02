# Banking77 hard-v1 · Tinker GPT-OSS-20B base rerun

This receipt is the first eval-set hill-climb from the 400-example seed-`20260901`
baseline. It retains all 65 examples that GPT-OSS-20B missed, drops the 335 examples
it passed, and replaces those passes with 335 unique examples sampled from the 2,680
held-out examples not used in round 0.

- Replacement seed: `20260902`
- Retained prior failures: `65`
- Removed prior passes: `335`
- Fresh replacements: `335`
- Total examples: `400`
- Parallelism: `50`
- Model: `openai/gpt-oss-20b`
- Temperature: `0.0`
- Reasoning effort: `low`
- Maximum completion: `1,024` tokens
- Banking77 container dataset digest: `sha256:9b52cfe49372fa6e6ce41654de86d641ac7fff56a89d1483b281f3e9732ff4dc`
- Exported held-out JSONL digest: `sha256:195988a7db018fb6bdee632f96f1e676047d7ebebf788a16afb1027b9e8d6fb4`

## Result

- Accuracy: `70.75%` (`283/400`)
- Macro-F1 across all 77 labels: `71.57%`
- Valid-label rate: `99.50%` (`398/400`)
- Retained-failure accuracy: `1.54%` (`1/65`)
- Fresh-replacement accuracy: `84.18%` (`282/335`)
- Provider request errors: `0`

The two invalid outputs comprise one analysis loop that exhausted the 1,024-token
ceiling and the same invented near-synonym `top_up_by_card` retained from round 0.
The complete selection lineage is pinned in
`../../manifests/banking77-hard-v1-seed-20260902.json`.

This is a base-model sampling run. No optimizer or training steps were submitted.
