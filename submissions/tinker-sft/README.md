# Tinker GPT-OSS-20B SFT

The reference SFT submission trains a rank-16 LoRA on Banking77's official training
split. A seeded stratified split reserves 10 examples per class for development.
Only assistant label tokens receive cross-entropy weight.

```bash
python scripts/train_tinker_banking77.py \
  --train-csv "$BANKING77_TRAIN_CSV" --env-file .env \
  --output-dir records/tinker/my-sft \
  --sft-updates 100 --sft-batch-size 64 --cispo-updates 0
```

Reference result (400-point development sample): **87.50% accuracy**, **86.29%
macro-F1**, and **100% valid labels**. The immutable challenge result is intentionally
not used for iteration.
