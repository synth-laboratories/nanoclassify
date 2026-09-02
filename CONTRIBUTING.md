# Contributing

Start with one falsifiable change. Keep challenge data sealed, use a deterministic
development split, and report failures as carefully as wins.

1. Fork the repository and create `submissions/<slug>/` from the template.
2. Run the base recipe and record its exact parent receipt.
3. Change one intervention at a time and evaluate on development data.
4. Run `pytest -q` and `python scripts/validate_public_release.py`.
5. Open a pull request containing code, aggregate metrics, and reproducibility
   metadata—but no private rows, gold labels, or per-example challenge predictions.

Tinker credentials belong in a project-local `.env` (gitignored). The reference
scripts read only `TINKER_API_KEY` and never print or copy it.
