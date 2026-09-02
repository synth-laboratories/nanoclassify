# Banking77 eval-set hill climb

Starting from a deterministic 400-example random baseline, each hardening round keeps
the examples GPT-OSS-20B missed and replaces its passes with a seeded sample from the
held-out examples never exposed in any earlier round. Every model run uses
`openai/gpt-oss-20b`, temperature `0.0`, low reasoning effort, a 1,024-token ceiling,
and parallelism `50`.

| Set | Replacement seed | Retained failures | Fresh replacements | Accuracy | Macro-F1 | Cumulative examples seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random baseline | 20260901 | — | 400 | 83.75% | 82.73% | 400 |
| hard-v1 | 20260902 | 65 | 335 | 70.75% | 71.57% | 735 |
| hard-v2 | 20260903 | 117 | 283 | 60.75% | 64.13% | 1,018 |
| hard-v3 | 20260904 | 157 | 243 | 48.50% | 54.36% | 1,261 |
| hard-v4 | 20260905 | 206 | 194 | 41.00% | 46.07% | 1,455 |
| hard-v5 | 20260906 | 236 | 164 | 35.25% | 40.10% | 1,619 |

## Added-round decomposition

| Set | Retained-failure accuracy | Fresh-replacement accuracy | Valid labels | Provider errors |
| --- | ---: | ---: | ---: | ---: |
| hard-v2 | 0.85% (1/117) | 85.51% (242/283) | 397/400 | 0 |
| hard-v3 | 0.64% (1/157) | 79.42% (193/243) | 397/400 | 0 |
| hard-v4 | 1.94% (4/206) | 82.47% (160/194) | 394/400 | 0 |
| hard-v5 | 1.27% (3/236) | 84.15% (138/164) | 394/400 | 0 |

The final challenge set contains 259 errors and 141 passes for this exact baseline.
Its low score is deliberately induced by selection and should not be reported as an
unbiased estimate of Banking77 generalization. The untouched pool still contains
1,461 of the 3,080 held-out examples.

Manifests pin every retained, removed, replacement, selected, and cumulatively seen
source index. Receipts contain all predictions, per-class metrics, confusion counts,
token usage, latency, and request errors.
