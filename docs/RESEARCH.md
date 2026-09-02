# Classification hill climbing: research writeup

## Thesis

Classification is a useful small-model research substrate because the output
contract is strict, evaluation is cheap and legible, and many model failures
can be localized to a boundary between a small number of labels. That makes it
possible to study optimization methods without conflating them with open-ended
generation quality.

NanoClassify treats each model or policy as a node in an evidence graph. A new
candidate must name its parent, the single intervention or bounded bundle that
created it, its training and inference configuration, and its measured result.
The objective is not to keep every experiment. It is to climb from a reproducible
baseline while preserving enough evidence to explain the path.

## Primary task: Banking77

Banking77 is the first task because it combines a large label space with many
semantically adjacent intents. Errors often reveal a specific boundary failure:
the model notices the topic but chooses the wrong intent among close neighbors.

The initial research sequence is:

1. Establish exact-label zero-shot and few-shot baselines.
2. Build label descriptions and confusion neighborhoods from training data only.
3. Categorize errors without reading sealed test labels during development.
4. Hill-climb prompts and inference procedures on a development split.
5. Compare SFT, rationale editing, hard-example replay, and verifier-guided
   candidate selection under matched budgets.
6. Evaluate promoted candidates once on the sealed held-out split.

## Transfer tasks

### ChemProt

ChemProt tests relation classification with biomedical terminology and a
different label ontology. It probes whether improvements arise from a general
classification method or Banking77-specific label engineering.

### DDI2013

DDI2013 tests closely related biomedical relation boundaries. Comparing it with
ChemProt should reveal whether the method transfers across similar domains with
different schemas and annotation conventions.

### LexGLUE SCOTUS

SCOTUS adds long-document legal context and issue-area classification. It
stresses context selection, truncation, and evidence localization rather than
short-utterance intent recognition.

## Candidate contract

Every candidate record should include:

- stable candidate ID and parent ID;
- task and dataset revision;
- immutable train, development, and held-out split hashes;
- base model and exact model revision;
- prompt/template hash;
- adapter or checkpoint identity when trained;
- decoding configuration;
- intervention description;
- training and inference usage;
- development and held-out metrics;
- per-class metrics and confusion data;
- invalid-output counts;
- artifact and reproduction paths;
- acceptance or rejection decision.

## Hill-climbing protocol

1. Start from a reproduced parent candidate.
2. State one hypothesis about a measurable failure mode.
3. Define a bounded intervention and maximum compute/spend.
4. Train or generate only from the permitted training partition.
5. Score on development data with the fixed evaluator.
6. Reject regressions and retain their receipts.
7. Promote only meaningful development improvements.
8. Use the sealed held-out split for promotion measurement, not iterative
   tuning.
9. Report variance across seeds or repeated samples where stochasticity matters.
10. Never replace exact task outcomes with an opaque shaped score.

## Metrics

Report at least:

- accuracy;
- macro F1;
- per-class precision, recall, F1, and support;
- confusion matrix;
- invalid or out-of-vocabulary output rate;
- abstention rate when supported;
- tokens, calls, wall time, and spend;
- confidence intervals or repeated-run variation;
- change relative to the reproduced parent.

Task-specific metrics may be added, but the common metrics should remain
available across all four tracks.

## Error taxonomy

Use a shared top-level taxonomy with task-specific refinements:

- label-boundary confusion;
- missing domain knowledge;
- ignored lexical cue;
- overweighted lexical cue;
- negation or scope error;
- entity/relation direction error;
- context-selection failure;
- truncation failure;
- ontology mismatch;
- unsupported assumption;
- format or exact-label failure;
- ambiguous or potentially mislabeled example.

Keep annotations separate from model outputs so the taxonomy can evolve without
silently changing the training corpus.

## Data integrity

- Pin dataset revisions and license metadata.
- Materialize deterministic split manifests.
- Hash every input partition.
- Prohibit held-out rows from prompts, synthetic-data requests, training, error
  mining, and candidate selection.
- Deduplicate exact and near-duplicate examples across partitions.
- Record every filter and rejection count.
- Do not commit restricted dataset contents when their licenses prohibit it.

## Research writeups

Each promoted candidate should produce a short writeup answering:

1. What failure did we target?
2. What changed relative to the parent?
3. What stayed controlled?
4. What did development and held-out metrics do?
5. Which classes improved or regressed?
6. What did the intervention cost?
7. What evidence supports the causal story?
8. What is the next falsifiable experiment?

## Initial milestone

The first complete milestone is a locally reproducible Banking77 baseline that
uses the existing Synth evaluation container, emits exact-label predictions,
produces per-example and aggregate receipts, and creates the root candidate from
which subsequent hill-climbing experiments descend.
