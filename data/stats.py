#!/usr/bin/env python
"""Thống kê dữ liệu: số mẫu mỗi tập, phân bố độ dài nhãn, tần suất ký tự, chiều rộng ảnh.
Xuất outputs/stats/<lang>_*.{md,csv,png}.
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_utils import nfc  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"  # hiển thị được chữ có dấu
ORDER = ["train", "val", "test", "hard", "train_clean"]


def read_split(d: Path):
    rows = []
    for line in (d / "labels.tsv").read_text(encoding="utf-8").split("\n"):
        if line:
            p = line.split("\t")
            rows.append((d / p[0], nfc(p[1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--lang", choices=["en", "vi", "both"], default="both")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    H, root = cfg["img_height"], Path(a.root or cfg["out_root"])
    root = root if root.is_absolute() else ROOT / root
    out = ROOT / "outputs" / "stats"
    out.mkdir(parents=True, exist_ok=True)

    for lang in (["en", "vi"] if a.lang == "both" else [a.lang]):
        ldir = root / lang
        if not ldir.exists():
            continue
        splits = [s for s in ORDER if (ldir / s / "labels.tsv").exists()]
        data = {s: read_split(ldir / s) for s in splits}

        # --- bảng tổng quan
        table = []
        for s in splits:
            lens = [len(t) for _, t in data[s]]
            widths = []
            for p, _ in data[s][:500]:  # lấy mẫu 500 ảnh để tiết kiệm thời gian
                w, h = Image.open(p).size
                widths.append(round(w * H / h))
            table.append([s, len(lens), len({t.casefold() for _, t in data[s]}), min(lens),
                          round(sum(lens) / len(lens), 1), max(lens), max(widths),
                          round(100 * sum(w > 768 for w in widths) / len(widths), 1)])
        hdr = ["tập", "số mẫu", "nhãn khác nhau", "len min", "len TB", "len max", f"rộng max @H={H}", "% rộng>768"]
        with open(out / f"{lang}_summary.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([hdr] + table)
        md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        md += ["| " + " | ".join(map(str, r)) + " |" for r in table]
        (out / f"{lang}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\n== {lang.upper()} ==\n" + "\n".join(md))

        # --- phân bố độ dài nhãn
        fig, ax = plt.subplots(figsize=(8, 4))
        for s in splits:
            ax.hist([len(t) for _, t in data[s]], bins=range(0, cfg["text"]["max_chars"] + 2),
                    alpha=0.5, label=s, density=True)
        ax.set(title=f"[{lang}] Phân bố độ dài nhãn", xlabel="số ký tự", ylabel="tỉ lệ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{lang}_label_length.png", dpi=130)
        plt.close(fig)

        # --- tần suất ký tự (tập train)
        cnt = collections.Counter("".join(t for _, t in data["train"]))
        with open(out / f"{lang}_char_freq.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["char", "codepoint", "count"])
            for c, n in cnt.most_common():
                w.writerow([c if c != " " else "<space>", f"U+{ord(c):04X}", n])
        top = cnt.most_common(70)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(top)), [n for _, n in top])
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([c if c != " " else "␣" for c, _ in top], fontsize=9)
        ax.set(title=f"[{lang}] Tần suất ký tự trong train (top {len(top)}/{len(cnt)} ký tự)", ylabel="số lần")
        fig.tight_layout()
        fig.savefig(out / f"{lang}_char_freq.png", dpi=130)
        plt.close(fig)
        rare = [c for c, n in cnt.items() if n < 20]
        print(f"[{lang}] {len(cnt)} ký tự khác nhau trong train; {len(rare)} ký tự hiếm (<20 lần): {''.join(sorted(rare))[:60]}")


if __name__ == "__main__":
    main()
