# Banking77 CISPO v5 — fixed seeds, 64 rollouts, 50 updates

This is the first fully separated CISPO campaign:

1. Load the immutable 156-seed qualifying manifest.
2. Generate 64 fresh training rollouts for every saved seed.
3. Run exactly 50 CISPO optimizer updates, four groups per update.

- Training rollouts: 9,984
- Unique seed groups: 156
- Group presentations: 200 (deterministic reshuffle/replay after the first pass)
- Learning rate: `1e-6`
- Development result: **87.75% accuracy**, **86.78% macro-F1**, **100% valid labels**
- SFT parent: **87.50% accuracy**, **86.29% macro-F1**

This improves on the SFT parent by 0.25 accuracy points and 0.49 macro-F1 points.
See [`train-reward.html`](train-reward.html) for the interactive reward trace.
