# nanoclassify

[![CI](https://github.com/synth-laboratories/nanoclassify/actions/workflows/ci.yml/badge.svg)](https://github.com/synth-laboratories/nanoclassify/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-dfff42.svg)](LICENSE)

Small-model classification research through reproducible hill climbing.

`nanoclassify` is a sister project to
[`nanohorizon`](https://github.com/synth-laboratories/nanohorizon). Where
NanoHorizon studies agent policies in Craftax, NanoClassify studies how far a
small language model can be pushed on exact-label classification under a fixed
data, compute, and evaluation budget.

The first task is **Banking77**. Three additional tracks test whether the same
research method transfers across scientific relation extraction, drug-drug
interaction classification, and legal decision classification:

| Track | Domain | Initial role |
|---|---|---|
| Banking77 | Customer-support intent classification | Primary hill-climbing task |
| ChemProt | Chemical-protein relation classification | Biomedical transfer |
| DDI2013 | Drug-drug interaction classification | Biomedical boundary stress test |
| LexGLUE SCOTUS | Supreme Court issue-area classification | Long-document legal transfer |

## Research question

Given a fixed base LM and bounded budget, which changes reliably improve
held-out classification accuracy without contaminating the test set or hiding
cost, variance, and invalid-output failures?

Candidate interventions include:

- prompt and label-description optimization;
- ontology and confusion-set construction;
- hard-example mining;
- synthetic rationales and error annotations;
- supervised fine-tuning;
- preference or verifier-guided hill climbing;
- curriculum and class-balancing strategies;
- compact inference-time decision procedures.

Every claimed improvement should be evaluated against the same immutable split,
exact-label parser, decoding configuration, and cost accounting.

## Getting started with LMs

The initial learning path is deliberately simple:

1. Run an exact-label zero-shot baseline.
2. Inspect the confusion matrix and invalid-output rate.
3. Propose one bounded change.
4. Evaluate it on development data.
5. Promote it only if a sealed held-out evaluation improves.
6. Record the candidate, parent, metrics, configuration, usage, and artifacts.

The project is not about producing one opaque training run. It is about making
the climb legible: what changed, why it was expected to help, what actually
moved, what regressed, and how much it cost.

See [the research writeup](docs/RESEARCH.md) for the experiment contract and
[the container references](docs/CONTAINERS.md) for existing task interfaces.
The canonical Banking77 challenge set is pinned by the
[heldout lock](tasks/banking77/HELDOUT.md).

## Try the challenge

The reference GPT-OSS-20B SFT reaches **87.50% accuracy** and **86.29% macro-F1**
on a fixed 400-point development sample. The leading separated CISPO continuation
reaches **87.75% accuracy** and **86.78% macro-F1**.

Start with [`submissions/tinker-sft`](submissions/tinker-sft/) or copy the
[`submission template`](submissions/template/). See [CONTRIBUTING.md](CONTRIBUTING.md)
for the no-leak submission contract. A static challenge page lives in [`site/`](site/).

The current development leader is **Tinker SFT + CISPO v5** at **87.75% accuracy**
and **86.78% macro-F1**. Development and sealed-heldout rankings are maintained
separately in [`records/leaderboard.json`](records/leaderboard.json).

## Quickstart

The repository itself has no mandatory runtime dependencies beyond Python 3.11+.

```bash
git clone https://github.com/synth-laboratories/nanoclassify.git
cd nanoclassify
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
pytest -q
python scripts/validate_public_release.py
```

Provider-backed recipes live under [`submissions/`](submissions/). Put
`TINKER_API_KEY` in a project-local `.env`; raw datasets and challenge-heldout
membership are intentionally not distributed.

## Repository layout

```text
docs/
  RESEARCH.md       research thesis, hill-climbing protocol, and reporting
  CONTAINERS.md     external container references and adapter boundaries
src/nanoclassify/   shared task, candidate, scoring, and receipt primitives
tasks/
  banking77/        primary task
  chemprot/         transfer task
  ddi2013/          transfer task
  lexglue_scotus/   transfer task
```

## Status

The Banking77 GPT-OSS-20B base baseline and five reproducible hard-example-mining
rounds are complete. The resulting 400-example hard-v5 set is locked as the canonical
challenge heldout; its mining baseline scores 35.25% accuracy and 40.10% macro-F1.
See [the hardening report](records/BANKING77_HARDENING.md) for full provenance.
