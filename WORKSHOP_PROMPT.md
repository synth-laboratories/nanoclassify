# Start a NanoClassify submission in Workshop

Paste this into Workshop from a checkout of the repository:

```text
Help me create a new NanoClassify submission.

First read README.md, CONTRIBUTING.md, submissions/README.md, records/leaderboard.json, and the closest existing submission recipe. Then propose and implement one bounded, interpretable intervention for the Banking77 development benchmark. Start from an existing public submission when possible and clearly name its parent.

Work only with the published development interface. Never request, reconstruct, inspect, or tune against heldout membership, heldout gold labels, or per-example heldout predictions. Do not run a sealed-heldout evaluation. Keep provider credentials in a project-local .env and use only credentials already configured there.

Create a complete submission directory from submissions/template. Record the model, parent, split digest, random seeds, decoding settings, training settings, aggregate cost and usage, accuracy, macro-F1, valid-label rate, failures, and artifact paths. Preserve the development artifacts needed to reproduce the aggregate result. Run the relevant tests and scripts/validate_public_release.py before finishing.

At the end, give me: (1) the hypothesis, (2) exactly what changed, (3) development results versus the parent, (4) cost and failure summary, (5) reproducibility commands, and (6) remaining risks. Prepare the submission for review, but do not publish, push, open a pull request, or promote it to heldout without asking me first.
```

The development leaderboard is the iteration surface. A sealed-heldout run is a separate promotion decision governed by the [submission contract](CONTRIBUTING.md).
