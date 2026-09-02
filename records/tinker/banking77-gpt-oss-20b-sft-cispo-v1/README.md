# Banking77 GPT-OSS-20B SFT + CISPO v1

- Parent: SFT v1
- Optimization: 12 updates, 32 prompts/update, 4 rollouts/prompt
- Sampling: 1,536 total trajectories, parallelism 50
- Development evaluation: same fixed 400-example sample as SFT
- Result: **87.00% accuracy**, **86.01% macro-F1**, **100% valid labels**

This is a valid negative result: CISPO regressed 0.50 accuracy points against its
parent on development and is not promoted as the best candidate. It remains a
reference implementation and reproducible submission.
