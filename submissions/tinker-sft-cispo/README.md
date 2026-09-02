# Tinker GPT-OSS-20B SFT + CISPO

This submission restores the SFT state with fresh optimizer moments and performs
native Tinker CISPO updates. Each update samples 32 independently seeded training
prompts with four rollouts each (128 trajectories, parallelism 50). Exact normalized
label correctness is the reward. Advantages are centered within each same-prompt
group and divided over trained completion tokens; all-identical groups are skipped.

The stricter v2 variant collects all rollouts before RL: eight rollouts for each
of 1,000 seeds, retaining only groups with 1–5 correct. It trained on 1,256 eligible
trajectories but reached 87.0% development accuracy, so it was not promoted.
Reducing the group minibatch to four yielded exactly 40 updates over 160 eligible
groups; that run also reached 87.0%, with 85.88% macro-F1.

The corrected v5 pipeline records 156 qualifying seed IDs once, then loads those
same seeds in a separate CISPO process. With 64 fresh rollouts per seed and 50
updates, it reaches **87.75% accuracy / 86.78% macro-F1**, beating its SFT parent.

```bash
python scripts/train_tinker_banking77.py \
  --train-csv "$BANKING77_TRAIN_CSV" --env-file .env \
  --output-dir records/tinker/my-cispo \
  --parent-sft-json records/tinker/my-sft/sft.json \
  --cispo-updates 12 --cispo-prompts-per-update 32 \
  --rollouts-per-prompt 4 --parallelism 50
```

For the up-front filtered variant, replace the CISPO arguments with:

```bash
  --cispo-upfront-prompts 1000 --rollouts-per-prompt 8 \
  --cispo-min-correct 1 --cispo-max-correct 5 \
  --cispo-groups-per-update 16 --cispo-learning-rate 1e-6 --parallelism 50
```
