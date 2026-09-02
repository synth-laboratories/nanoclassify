# Banking77 SFT + up-front filtered CISPO v2

Exactly eight rollouts were collected for each of 1,000 independently selected
training seeds from the fixed SFT parent checkpoint. Only groups with 1–5 correct
rollouts entered CISPO; no intermediate policy was used for resampling.

- Total up-front rollouts: 8,000
- Correctness histogram (0 through 8): `80, 26, 30, 33, 33, 35, 72, 134, 557`
- Eligible groups: 157
- Eligible trajectories trained: 1,256
- Training: one shuffled pass, 16 groups/update, 10 updates, learning rate `1e-6`
- Development result: **87.00% accuracy**, **85.91% macro-F1**, **100% valid labels**
- Parent SFT result: **87.50% accuracy**, **86.29% macro-F1**

This candidate was not promoted to sealed heldout evaluation because it regressed
against its parent on the fixed development sample.
