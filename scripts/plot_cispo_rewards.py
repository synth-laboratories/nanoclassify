#!/usr/bin/env python3
"""Plot offline CISPO training-batch rewards from a campaign receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rollouts = int(receipt["collection"]["rollouts_per_prompt"])
    history = receipt["history"]
    updates = [int(row["update"]) for row in history]
    rewards = [sum(row["correct_counts"]) / (len(row["correct_counts"]) * rollouts)
               for row in history]
    cumulative = [sum(rewards[:index]) / index for index in range(1, len(rewards) + 1)]

    width, height = 1800, 990
    image = Image.new("RGB", (width, height), "#f3f0e7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    title_font = ImageFont.load_default(size=38)
    left, top, right, bottom = 150, 145, 1720, 790
    y_max = 0.36

    def point(update: int, reward: float) -> tuple[int, int]:
        x = left + int((update - 1) / max(1, len(updates) - 1) * (right - left))
        y = bottom - int(reward / y_max * (bottom - top))
        return x, y

    band_top = point(1, 5 / rollouts)[1]
    band_bottom = point(1, 1 / rollouts)[1]
    draw.rectangle((left, band_top, right, band_bottom), fill="#e7edb0")
    for tick in range(0, 7):
        value = tick * 0.05
        y = point(1, value)[1]
        draw.line((left, y, right, y), fill="#cbc8ba", width=2)
        draw.text((55, y - 12), f"{value:.2f}", fill="#55554d", font=small)
    draw.line((left, top, left, bottom), fill="#171712", width=3)
    draw.line((left, bottom, right, bottom), fill="#171712", width=3)
    for update in (1, 5, 10, 15, 20, len(updates)):
        if update <= len(updates):
            x, _ = point(update, 0)
            draw.text((x - 10, bottom + 20), str(update), fill="#55554d", font=small)
    reward_points = [point(update, reward) for update, reward in zip(updates, rewards, strict=True)]
    mean_points = [point(update, reward) for update, reward in zip(updates, cumulative, strict=True)]
    draw.line(reward_points, fill="#788600", width=5, joint="curve")
    for x, y in reward_points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#788600")
    draw.line(mean_points, fill="#171712", width=8, joint="curve")
    draw.text((left, 55), "CISPO offline training reward - 16 rollouts per group",
              fill="#171712", font=title_font)
    draw.text((left, 95), "Exact-label reward in each shuffled optimizer minibatch",
              fill="#55554d", font=font)
    draw.text((760, 880), "optimizer update", fill="#171712", font=font)
    draw.text((left, 835),
              "Rewards were sampled once from the SFT parent before training; this is not an on-policy learning curve.",
              fill="#55554d", font=small)
    draw.line((1180, 75, 1250, 75), fill="#788600", width=5)
    draw.text((1265, 62), "batch mean", fill="#171712", font=small)
    draw.line((1180, 110, 1250, 110), fill="#171712", width=8)
    draw.text((1265, 97), "cumulative mean", fill="#171712", font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
