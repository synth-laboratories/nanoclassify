# NanoClassify submissions

[![CI](https://github.com/synth-laboratories/nanoclassify/actions/workflows/ci.yml/badge.svg)](https://github.com/synth-laboratories/nanoclassify/actions/workflows/ci.yml) [![Website](https://img.shields.io/badge/website-leaderboard-7ee0a8.svg)](https://synth-laboratories.github.io/nanoclassify/)

Reproducible classification hill climbing from Synth Laboratories. Development and sealed-heldout results are ranked separately.

## Development · Banking77 · 400 examples

| # | Submission | Model | Accuracy | Macro-F1 | Recipe |
|---:|---|---|---:|---:|---|
| 1 | **Tinker SFT + CISPO v5** | GPT-OSS-20B | **87.75%** | **86.78%** | [64 rollouts · 50 updates](records/tinker/banking77-gpt-oss-20b-cispo-v5-64r-50u/) |
| 2 | Tinker SFT v1 | GPT-OSS-20B | 87.50% | 86.29% | [rank-16 LoRA](records/tinker/banking77-gpt-oss-20b-sft-v1/) |

## Sealed heldout · Banking77 · 400 examples

| # | Submission | Model | Accuracy | Macro-F1 | Receipt |
|---:|---|---|---:|---:|---|
| 1 | **Tinker SFT v1** | GPT-OSS-20B | **51.25%** | **53.95%** | [record](records/tinker/banking77-gpt-oss-20b-sft-v1/official-heldout-400/) |
| 2 | GEPA large v5 frontier | GPT-OSS-20B | 44.75% | 48.19% | [record](records/banking77-gepa-oss20b-large-v5-heldout/) |
| 3 | Base | GPT-OSS-20B | 35.25% | 40.10% | [record](records/banking77-hard-v5-tinker-gpt-oss-20b-base-400-seed-20260906/) |

**Ready to climb?** Copy the [Workshop starter prompt](WORKSHOP_PROMPT.md), begin from the [submission template](submissions/template/), and follow the [submission contract](CONTRIBUTING.md).

<details>
<summary>Project notes and local validation</summary>

NanoClassify is a sister project to [NanoHorizon](https://github.com/synth-laboratories/nanohorizon). Banking77 is the primary task; ChemProt, DDI2013, and LexGLUE SCOTUS are transfer tracks. See [the research writeup](docs/RESEARCH.md).

```bash
git clone https://github.com/synth-laboratories/nanoclassify.git
cd nanoclassify
python -m venv .venv && source .venv/bin/activate
python -m pip install -e . pytest
pytest -q && python scripts/validate_public_release.py
```

Keep provider credentials in a project-local `.env`. Raw datasets, heldout membership, gold labels, and per-example heldout predictions are intentionally not distributed.
</details>
