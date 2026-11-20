#!/usr/bin/env python
"""Vẽ lưới ảnh mẫu kèm nhãn (qua Dataset, tức đúng ảnh mà mô hình sẽ thấy sau tiền xử lý)."""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import OCRDataset  # noqa: E402
from src.text_utils import Charset  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--lang", choices=["en", "vi", "both"], default="both")
    ap.add_argument("--splits", nargs="+", default=["train", "hard"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=None, help="mặc định = img_height trong config")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(cfg["out_root"])
    root = root if root.is_absolute() else ROOT / root
    cs = Charset.load(ROOT / "data" / "charset.txt")
    H = a.height or cfg["img_height"]
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    for lang in (["en", "vi"] if a.lang == "both" else [a.lang]):
        for split in a.splits:
            d = root / lang / split
            if not (d / "labels.tsv").exists():
                continue
            ds = OCRDataset(d, cs, img_height=H)
            idx = random.Random(a.seed).sample(range(len(ds)), min(a.n, len(ds)))
            cols = 2
            rows = (len(idx) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(11, 0.95 * rows + 0.6))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, i in zip(axes.ravel(), idx):
                s = ds[i]
                ax.imshow(s["image"][0].numpy(), cmap="gray", vmin=-1, vmax=1, aspect="equal")
                ax.set_title(s["text"], fontsize=10, loc="left", pad=2)
            fig.suptitle(f"[{lang}/{split}] {len(idx)} ảnh mẫu (sau tiền xử lý, H={H})", fontsize=11)
            fig.tight_layout()
            p = out / f"samples_{lang}_{split}.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            print("Đã lưu", p)


if __name__ == "__main__":
    main()
