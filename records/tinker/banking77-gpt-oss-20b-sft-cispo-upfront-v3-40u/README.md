# Banking77 up-front filtered CISPO v3 — 40 updates

This repeats the fixed-parent collection protocol with smaller RL minibatches to
produce exactly 40 optimizer updates.

- Seeds: 1,000
- Rollouts per seed: 8
- Total rollouts: 8,000
- Eligible range: 1–5 correct out of 8
- Eligible groups: 160
- Eligible trajectories: 1,280
- Training: one shuffled pass, 4 groups/update, exactly 40 updates
- Learning rate: `1e-6`
- Development result: **87.00% accuracy**, **85.88% macro-F1**, **100% valid labels**
- Parent SFT: **87.50% accuracy**, **86.29% macro-F1**

The longer optimizer schedule did not improve accuracy and slightly reduced macro-F1.
It was therefore not promoted to sealed heldout evaluation.
