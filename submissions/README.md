# Submissions

A submission is a reproducible training recipe plus a public receipt. It must
identify its parent, data split digest, model, seed, hyperparameters, checkpoint,
development metrics, and provider usage. Never commit challenge queries, labels,
per-example predictions, API keys, or private dataset files.

Current reference recipes:

- [`tinker-sft`](tinker-sft/): GPT-OSS-20B LoRA supervised fine-tuning.
- [`tinker-sft-cispo`](tinker-sft-cispo/): native CISPO continued from the SFT state.

Copy [`template`](template/) for a new entry. Run `python scripts/validate_public_release.py`
before opening a pull request. Challenge-heldout scoring is performed by maintainers;
contributors develop against their own deterministic training split.
