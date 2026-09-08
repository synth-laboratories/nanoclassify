# Common Nano entrypoints

This repository vendors **nano-standard 1.0.0**, a dependency-free Python 3.11+
CLI. `nano.py` checks the bundle against `tools/nano-standard.lock.json` before
loading it. The shared source project is `nano-standard`; clones need no sibling
checkout. `nano.json` is this repository's explicit adapter/track/lane registry.

```bash
uv run --no-project python nano.py doctor
uv run --no-project python nano.py --help
```

`uv run --no-project` runs the standard-library CLI without resolving the Nano
project's dependencies or changing its lockfile. Use Python 3.11 or newer. For an
installed interpreter with no network access, use:

```bash
uv run --offline --no-python-downloads --no-project --python python3 python nano.py doctor
```

All commands resolve the benchmark root from the launcher. Run relative input
and output paths from the repository root. Use `NANO_PYTHON=/path/to/python` for
the native evaluator's dependency environment; the shared CLI itself uses stdlib.

## Vocabulary and responsibilities

Benchmark = task family; track = editable rights; lane = compute entitlement;
rung = difficulty/criterion; split = data use; arm = experimental condition;
recipe = method. Only listed track/lane pairs are supported. `doctor` lists them.

Shared code packages explicit source files, freezes plans, checks case identities,
measured resource fields and trace hashes, and records process execution.
Native evaluators retain task scoring, per-request budget enforcement, trusted
runtime isolation, artifact loading, domain eligibility and ranking.

The installed adapters are **development adapters**. Shared verification is not
an independent attestation of self-reported usage or trace truth. It cannot
promote a result onto an official board. Existing native official protocols
retain their own admission paths. `mode=official` is rejected here until an audited
common adapter is admitted. No historical result is relabeled.

## Set up the native runtime with uv

Contract checks, packaging and previews need only the commands above. Before
executing training or an evaluator, prepare that repository's environment from
its root:

| Repository | Setup | Additional runtime requirements |
|---|---|---|
| NanoHorizon | `uv sync --locked` | Native Craftax facade and the selected MLX/provider runtime; follow README and TRAIN_EVAL. The MLX runtime is a separate sibling environment. |
| NanoCoop | `uv sync --locked` | Native DungeonGrid runtime and a valid domain plan. |
| NanoAlign | `uv sync --locked --extra dev` | Runtime/model/provider setup for the selected track; this includes local test dependencies. |
| NanoClassify | `uv sync` | This checkout has no uv.lock yet; review the generated lock before committing. Install the selected provider runtime as documented by its native evaluator. |
| NanoProgram | `uv sync --locked` | Python 3.14+ and the local `../synth-lab` dependency declared in pyproject.toml, plus the selected task containers. |

These commands install declared project dependencies. They do not provision
models, containers, provider credentials or optional runtimes omitted from the
project manifest. A locked sync that fails because source and lock disagree is a
reconciliation task: review the dependency change before updating the lock.

Once prepared, `uv run --no-sync python nano.py ...` uses the project interpreter
without resynchronizing it. Python-based native adapters inherit that interpreter;
`NANO_PYTHON` can explicitly select a different prepared environment. Shell-based
adapters retain their native interpreter-selection logic: check the printed
command and the domain setup instructions, particularly NanoHorizon's MLX setup.
Keep credentials in already-authorized local environment/.env flows; the shared
CLI does not load a .env automatically.

## Submission workflow

Pick track, lane, artifact type and adapter from `doctor`:

| Repo / track | `--track` | `--artifact-type` | `--adapter` |
|---|---|---|---|
| NanoHorizon | `policy-improvement` | `policy` or `mlx-lora` | `craftax` |
| NanoCoop monitor | `monitor_steer` | `monitor` | `monitor-steer` |
| NanoCoop RL | `cooperative_rl` | `policy` | `cooperative-rl` |
| NanoAlign 1 | `track1` | `policy` | `track1-transfer` |
| NanoAlign 2 | `track2` | `monitor` | `track2-native` |
| NanoAlign 3 | `track3` | `policy` | `track3-fixture` |
| NanoAlign 4 | `track4` | `policy` | `track4-actual` |
| NanoClassify | `classification` | `checkpoint` or `policy` | `banking77` |
| NanoProgram | `prompt-optimization` | `optimizer` | `optimizer` |

All rows support `--lane local-development`. Use the values for your repo in the
following Horizon example. The scaffold is not runnable until you implement the
native candidate interface:

```bash
uv run --no-project python nano.py init-submission submissions/my-method --track policy-improvement --lane local-development --artifact-type policy
# Implement candidate.py and train.py; edit submission.json entrypoints,
# explicit file list and provenance to match the native domain interface.
uv run --no-project python nano.py package submissions/my-method/submission.json
uv run --no-project python nano.py validate submissions/my-method/submission.packaged.json
uv run --no-project python nano.py init-protocol --submission submissions/my-method/submission.packaged.json --adapter craftax --output development-protocol.json
```

Templates intentionally fail until implemented. `train.py` accepts `--out`;
the candidate must implement the native domain API, which differs by benchmark.
Training entrypoints are optional for inference-only/optimizer submissions: edit
the draft manifest explicitly. Package all runtime source and required assets by
listing them in `files`; packaging hashes only those files and never crawls
credentials, caches or unrelated working-tree changes. Include base model, data
pins, parent checkpoint and training receipt under `provenance` before admission.
An immutable package is never overwritten: use a new submission directory for a
new candidate or package revision.

## Plan and execution

The generated protocol is a development example with one dummy case and
`adapter_args=["--help"]`. Replace the cases with your actual domain plan,
including seed, arm, rung, repeat; provide exact native arguments, immutable pins,
run-level `limits` and per-case `case_limits`. `train_args` holds extra trainer
arguments. Native arguments must agree with the shared plan; development dispatch
does not translate arbitrary domain plans or prove that a native evaluator obeyed
the declared case allocation.

```bash
uv run --no-project python nano.py plan --submission submissions/my-method/submission.packaged.json --protocol development-protocol.json --output plan.json
uv run --no-project python nano.py train --submission submissions/my-method/submission.packaged.json --plan plan.json --output results/train-001
uv run --no-project python nano.py evaluate --submission submissions/my-method/submission.packaged.json --plan plan.json --output results/eval-001
```

These print the exact command without running it. After implementing the
candidate, replacing the protocol's dummy case/`--help` arguments, freezing the
plan and preparing the native environment, execute with:

```bash
uv run --no-sync python nano.py train --submission submissions/my-method/submission.packaged.json --plan plan.json --output results/train-001 --execute
uv run --no-sync python nano.py evaluate --submission submissions/my-method/submission.packaged.json --plan plan.json --output results/eval-001 --execute
```

Use new output directories for each attempt; existing outputs are never
overwritten. An inference-only or optimizer submission can omit the train step.
The plan must name the candidate artifact actually passed to the native evaluator;
when training produces a new immutable candidate, package that output and freeze
its evaluation plan before evaluation.

`--execute` invokes the frozen command. That may incur the native provider's charges: use the domain's
normal authorized budget workflow. Common execution imposes a process-group wall
timeout; dollar/token admission stays in the native evaluator. It inherits the
existing environment and never reads Keychain or loads secrets automatically.
Remote jobs require their native cleanup mechanism; killing a local process does
not guarantee cancellation of provider jobs.

Execution creates a new directory with `plan.json`, `process.log` and
`execution.json`. A successful process is not a scored result. `NANO_PLAN`,
`NANO_SUBMISSION`, and `NANO_OUTPUT` are supplied to adapters. Native output and
candidate paths must be supplied in `adapter_args` when the native command needs
them; the common output directory does not redirect arbitrary native artifacts.
Never put credential values in command arguments or public receipt bundles.

## Evidence verification

```bash
uv run --no-project python nano.py result-template --plan plan.json --output result.json
# The native evaluator/exporter populates the result with measured evidence.
uv run --no-project python nano.py verify --plan plan.json --result result.json --evidence-root results/eval-001
```

Each case result needs `id`, `case_digest` (canonical SHA-256 of the planned
case), `status` (completed/censored/failed), `termination`, numeric-or-null
`metrics`, measured `usage` for declared case limits, and `trace` containing
relative `path` and file `sha256`. Run usage must include every declared run limit
with matching units. Hashes use UTF-8 JSON sorted keys, compact separators and no
NaN (`nano_standard.core.digest`).

Unknown/duplicate cases, wrong plans, changed traces, escaping paths, nonfinite
metrics, missing accounting and exceeded caps are invalid. Missing, censored or
failed cases produce an incomplete result and exit 2. Early domain termination
(e.g. legitimate death) is a completed case with an explicit termination reason;
an infrastructure failure is failed. The shared checker never recomputes a
scientific scalar from partial rows and always leaves ranking to native admission.

## Native adapters

- NanoHorizon: `craftax` → `eval_submission.sh`; supply submission slug and exact infer arguments. Public fixed seeds are legacy evaluation, not a new sealed test.
- NanoCoop: `monitor-steer` / `cooperative-rl` → `eval_submission.sh`; supply native `--plan`, candidate and output flags. Ranking remains disabled by domain rules.
- NanoAlign: `track1-transfer`, `track2-native`, `track3-fixture`, `track4-actual` → matching Python evaluator modules. Track 3 fixture is explicitly developmental. Preserve native protocol-scoped attestation and ranking.
- NanoClassify: `banking77` → `scripts/eval_tinker_banking77.py`; supply explicit data, output and provider configuration. Keep labels/private examples private.
- NanoProgram: `optimizer` → `evaluate_optimizer.sh`; supply optimizer file and container names. Optimizers do not require a weight-training phase.

## Compatibility and changes

Changing `nano.json` invalidates old package/protocol bindings for this checkout.
Keep the old registry with archived evidence; repackage only as a deliberate
protocol migration. Shared schemas and canonical semantics are documented in the
source project's `CONTRACTS.md`. Use `validate-ladder` for `nano.ladder.v1` files;
ladder promotion criteria are owned by domain evaluators.

Bundle updates must be built from reviewed shared source, tested, then installed
with a matching lock. Do not edit one vendored copy independently.
