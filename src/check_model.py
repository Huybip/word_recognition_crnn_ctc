#!/usr/bin/env python
"""Nghiệm thu Giai đoạn 2:
  1) in summary (shape từng tầng, số tham số);
  2) forward 1 batch giả + CTCLoss + backward: không NaN/Inf, gradient hữu hạn;
  3) kiểm tra T >= 2*Lmax+1 (và điều kiện chính xác T >= L+R) trên dữ liệu thật.
Chạy: python -m src.check_model [--height 48] [--splits train val test hard]
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .ctc_check import collect_stats, ctc_min_length, feasibility_report, worst_samples
from .model import build_model, summary
from .text_utils import Charset

ROOT = Path(__file__).resolve().parents[1]


def fake_batch(model, charset, n=4, h=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = h or model.img_height
    widths = torch.tensor([320, 256, 200, 128][:n])
    images = torch.randn(n, 1, h, int(widths.max()), generator=g)
    lens = torch.tensor([10, 7, 5, 3][:n])
    targets = torch.randint(1, charset.num_classes, (int(lens.sum()),), generator=g)
    return images, targets, lens, widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--data_config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--charset", default=str(ROOT / "data" / "charset.txt"))
    ap.add_argument("--height", type=int, default=None, help="ghi đè img_height")
    ap.add_argument("--no_rnn", action="store_true", help="kiểm tra biến thể CNN + Linear")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "hard"])
    ap.add_argument("--langs", nargs="+", default=["en", "vi"])
    a = ap.parse_args()

    mcfg = yaml.safe_load(open(a.model_config, encoding="utf-8"))
    dcfg = yaml.safe_load(open(a.data_config, encoding="utf-8"))
    if a.height:
        mcfg["img_height"] = a.height
    if a.no_rnn:
        mcfg["rnn"]["use_rnn"] = False
    cs = Charset.load(a.charset)
    model = build_model(cs.num_classes, mcfg)

    print("=" * 20, f"1) SUMMARY (num_classes={cs.num_classes} = {len(cs)} ký tự + blank)", "=" * 20)
    print(summary(model, width=320))

    print("\n" + "=" * 20, "2) FORWARD + CTC + BACKWARD trên batch giả", "=" * 20)
    images, targets, tl, widths = fake_batch(model, cs)
    model.train()
    out = model(images, widths)
    il = model.output_lengths(widths)
    print(f"images {tuple(images.shape)} -> log_probs {tuple(out.shape)} (T,N,C) | input_lengths={il.tolist()} target_lengths={tl.tolist()}")
    print(f"tổng exp(log_probs) theo C (phải =1): {out.exp().sum(2).mean():.6f}")
    loss = nn.CTCLoss(blank=0, zero_infinity=False)(out, targets, il, tl)  # zero_infinity=False để lộ mẫu vô nghiệm
    loss.backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"CTCLoss = {loss.item():.4f} | hữu hạn: {bool(torch.isfinite(loss))} | gradient hữu hạn: {gfin}")
    assert torch.isfinite(loss) and gfin, "Loss/gradient không hữu hạn!"

    print("\n" + "=" * 20, f"3) KIỂM TRA T so với nhãn (img_height={mcfg['img_height']})", "=" * 20)
    root = Path(dcfg["out_root"])
    root = root if root.is_absolute() else ROOT / root
    dirs = [root / l / s for l in a.langs for s in a.splits if (root / l / s / "labels.tsv").exists()]
    if not dirs:
        print("Chưa có dữ liệu để kiểm tra (chạy Giai đoạn 1).")
        return
    stats = collect_stats(dirs, cs, mcfg["img_height"])
    text, res, best = feasibility_report(stats, strides=(4, 2, 1), img_height=mcfg["img_height"])
    print(text)
    s = model.total_stride
    print(f"\nCấu hình hiện tại: total_stride = {s}")
    for sp, t, w in worst_samples(stats, s):
        print(f"  mẫu rủi ro nhất: {sp:6s} W={w:4d} T={w // s:3d} cần>={ctc_min_length(t):2d} (L={len(t)}) {t!r}")


if __name__ == "__main__":
    main()
