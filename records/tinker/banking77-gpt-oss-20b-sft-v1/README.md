# Banking77 GPT-OSS-20B SFT v1

- Base: `openai/gpt-oss-20b`
- Adapter: rank-16 Tinker LoRA
- Training: 100 updates × 64 examples, learning rate `2e-5`
- Split: seed `20260907`, 10 development examples per class
- Development evaluation: 400 examples, seed `20260908`, parallelism 50
- Result: **87.50% accuracy**, **86.29% macro-F1**, **100% valid labels**
- Sealed hard-v5 result: **51.25% accuracy**, **53.95% macro-F1**, **100% valid labels**
- Uplift: **+16.00 accuracy points** over base; **+6.50** over the best GEPA prompt

The sampler and restorable state URLs are recorded in `sft.json`; both have a 30-day
TTL from creation. Per-example development predictions are deliberately untracked.
