#!/usr/bin/env python
"""Vẽ đường cong train/val loss, val CER, val word accuracy từ outputs/runs/<run>/metrics.csv.
Ví dụ: python -m src.plot_curves overfit200_en overfit200_vi_ft overfit200_vi_scratch --out outputs/figures/overfit_curves.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read(run):
    with open(ROOT / "outputs" / "runs" / run / "metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "figures" / "training_curves.png"))
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for run in a.runs:
        m = read(run)
        ax[0].plot(m["epoch"], m["train_loss"], label=run)
        ax[1].plot(m["epoch"], [100 * v for v in m["val_cer"]], label=run)
        ax[2].plot(m["epoch"], [100 * v for v in m["val_word_acc"]], label=run)
    for axis, (t, y) in zip(ax, [("Train loss (CTC)", "loss"), ("Val CER", "CER (%)"), ("Val Word Accuracy", "Word Acc (%)")]):
        axis.set(title=t, xlabel="epoch", ylabel=y)
        axis.grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[1].set_ylim(0, 105)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("Đã lưu", a.out)


if __name__ == "__main__":
    main()
