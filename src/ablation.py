#!/usr/bin/env python
"""Ablation tự động: huấn luyện từng biến thể -> đánh giá (greedy/beam) -> bảng tổng hợp.

Các thí nghiệm (tên viết tắt dùng trong --only / --alias):
  base_en     EN, đầy đủ (CNN + BiLSTM, có augmentation)           -> mốc so sánh
  norn_en     EN, use_rnn=false (chỉ CNN + Linear)                 -> ablation "có/không BiLSTM"
  aug_en      EN, có augmentation, CÙNG số mẫu với noaug_en         -> ablation "có/không augmentation"
  noaug_en    EN, huấn luyện trên train_clean (không augmentation)
  ft_vi       VI, fine-tune từ base_en (mở rộng lớp cuối)           -> ablation "fine-tune vs từ đầu"
  scratch_vi  VI, train từ đầu
  (greedy vs beam search lấy từ cột decoder của mọi lần đánh giá)
Yêu cầu train_clean: đặt counts.en.train_clean > 0 trong configs/data.yaml rồi sinh lại dữ liệu EN.

Ví dụ:
  python -m src.ablation --mode smoke --epochs 4                       # kiểm tra pipeline trên CPU với bộ smoke (số liệu VÔ NGHĨA)
  python -m src.ablation --mode full --alias base_en=en_base ft_vi=vi_finetune scratch_vi=vi_scratch --reuse
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from .eval import evaluate_sets, default_sets, load_checkpoint, save_outputs
from .train import load_config

ROOT = Path(__file__).resolve().parents[1]
ALL = ["base_en", "norn_en", "aug_en", "noaug_en", "ft_vi", "scratch_vi"]


def fmt(v) -> str:
    return "true" if v is True else "false" if v is False else "null" if v is None else (json.dumps(v) if isinstance(v, list) else str(v))


def n_lines(p: Path) -> int:
    return sum(1 for l in p.read_text(encoding="utf-8").split("\n") if l) if p.exists() else 0


def plan(tag: str, data_root: Path, lr_ft, aliases: dict, only):
    """Trả {exp: dict(run, lang, overrides, eval_sets)}; bỏ qua thí nghiệm thiếu dữ liệu."""
    run = lambda e: aliases.get(e, f"ab_{tag}_{e}")  # noqa: E731
    n_train, n_clean = n_lines(data_root / "en/train/labels.tsv"), n_lines(data_root / "en/train_clean/labels.tsv")
    vi = {"data.lang": "vi", "data.charset": "data/charset.txt"}
    P = {
        "base_en": dict(lang="en", overrides={}, sets=["en/test", "en/hard"]),
        "norn_en": dict(lang="en", overrides={"model.rnn.use_rnn": False}, sets=["en/test", "en/hard"]),
        "ft_vi": dict(lang="vi", overrides={**vi, "init_from": f"outputs/runs/{run('base_en')}/best.pt",
                                            **({"optim.lr": lr_ft} if lr_ft else {})}, sets=["vi/test", "vi/hard"]),
        "scratch_vi": dict(lang="vi", overrides={**vi, "init_from": None}, sets=["vi/test", "vi/hard"]),
    }
    if n_clean:
        P["noaug_en"] = dict(lang="en", overrides={"data.train_split": "train_clean"}, sets=["en/test", "en/hard"])
        P["aug_en"] = (dict(lang="en", overrides={"data.max_train_samples": n_clean}, sets=["en/test", "en/hard"])
                       if n_clean < n_train else dict(alias_of="base_en"))
    else:
        print("(!) Không có en/train_clean -> bỏ qua ablation augmentation. Đặt counts.en.train_clean > 0 và sinh lại dữ liệu EN.")
    out = {}
    for e in ALL:
        if e not in P or (only and e not in only):
            continue
        spec = P[e]
        if "alias_of" in spec:
            out[e] = dict(run=run(spec["alias_of"]), alias_of=spec["alias_of"], lang="en", overrides={}, sets=["en/test", "en/hard"])
        else:
            out[e] = dict(run=run(e), **spec)
    return out


def train_cmd(cfg_path, spec, epochs):
    ov = {"run.name": spec["run"], **spec["overrides"]}
    if epochs:
        ov["train.epochs"] = epochs
    return [sys.executable, "-m", "src.train", "--config", str(cfg_path), "--overwrite", "--set"] + [f"{k}={fmt(v)}" for k, v in ov.items()]


def cell(res, exp, sset, dec, key, scale=100.0, nd=2):
    for r in res.get(exp, {}).get("rows", []):
        if r["set"] == sset and r["decoder"] == dec:
            return f"{r[key] * scale:.{nd}f}"
    return "—"


def build_summary(res: dict, decoders: list) -> str:
    """Bảng tổng hợp 4 câu hỏi ablation từ res[exp] = dict(rows=[...], summary={...})."""
    g, L = decoders[0], []
    sm = lambda e, k: res.get(e, {}).get("summary", {}).get(k, "—")  # noqa: E731
    L += ["## Ablation 1: có / không BiLSTM (EN, greedy)", "",
          "| Biến thể | Tham số | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) |", "|---|---:|---:|---:|---:|---:|"]
    for e, name in [("base_en", "CNN + BiLSTM + Linear"), ("norn_en", "CNN + Linear (không RNN)")]:
        L.append(f"| {name} | {sm(e, 'params'):,} | {cell(res, e, 'en_test', g, 'cer')} | {cell(res, e, 'en_test', g, 'word_acc')} | "
                 f"{cell(res, e, 'en_hard', g, 'cer')} | {cell(res, e, 'en_hard', g, 'word_acc')} |" if isinstance(sm(e, 'params'), int) else
                 f"| {name} | — | {cell(res, e, 'en_test', g, 'cer')} | {cell(res, e, 'en_test', g, 'word_acc')} | — | — |")
    L += ["", "## Ablation 2: có / không augmentation (EN, greedy, cùng số mẫu huấn luyện)", "",
          "| Biến thể | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) |", "|---|---:|---:|---:|---:|"]
    for e, name in [("aug_en", "có augmentation (train)"), ("noaug_en", "không augmentation (train_clean)")]:
        L.append(f"| {name} | {cell(res, e, 'en_test', g, 'cer')} | {cell(res, e, 'en_test', g, 'word_acc')} | "
                 f"{cell(res, e, 'en_hard', g, 'cer')} | {cell(res, e, 'en_hard', g, 'word_acc')} |")
    L += ["", f"## Ablation 3: greedy vs beam search (mô hình base_en)", "",
          "| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh (forward+decode) | ms decode |", "|---|---|---:|---:|---:|---:|"]
    for s in ("en_test", "en_hard"):
        for d in decoders:
            L.append(f"| {s} | {d} | {cell(res, 'base_en', s, d, 'cer')} | {cell(res, 'base_en', s, d, 'word_acc')} | "
                     f"{cell(res, 'base_en', s, d, 'ms_total', 1, 2)} | {cell(res, 'base_en', s, d, 'ms_decode', 1, 2)} |")
    L += ["", "## Ablation 4: fine-tune từ EN vs train VI từ đầu (greedy)", "",
          "| Biến thể | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) | epoch tốt nhất | val CER tốt nhất (%) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for e, name in [("ft_vi", "fine-tune từ EN"), ("scratch_vi", "từ đầu")]:
        b = sm(e, "best_val_cer")
        L.append(f"| {name} | {cell(res, e, 'vi_test', g, 'cer')} | {cell(res, e, 'vi_test', g, 'word_acc')} | "
                 f"{cell(res, e, 'vi_hard', g, 'cer')} | {cell(res, e, 'vi_hard', g, 'word_acc')} | {sm(e, 'best_epoch')} | "
                 f"{'—' if b == '—' else f'{100 * b:.2f}'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--config", default=None, help="mặc định: small_cpu.yaml (smoke) / full_gpu.yaml (full)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr_ft", type=float, default=None, help="learning rate khi fine-tune VI (mặc định: lr/3 của config)")
    ap.add_argument("--decoders", nargs="*", default=["greedy", "beam5", "beam20"])
    ap.add_argument("--only", nargs="*", default=None, choices=ALL)
    ap.add_argument("--alias", nargs="*", default=[], help="dùng lại run có sẵn, vd: base_en=en_base")
    ap.add_argument("--reuse", action="store_true", help="bỏ qua huấn luyện nếu best.pt đã tồn tại")
    ap.add_argument("--max_samples", type=int, default=None, help="giới hạn số mẫu mỗi tập khi đánh giá")
    ap.add_argument("--root", default="data/synth")
    a = ap.parse_args()

    cfg_path = ROOT / (a.config or ("configs/small_cpu.yaml" if a.mode == "smoke" else "configs/full_gpu.yaml"))
    base_lr = load_config(cfg_path)["optim"]["lr"]
    data_root = ROOT / a.root
    aliases = dict(x.split("=", 1) for x in a.alias)
    experiments = plan(a.mode, data_root, a.lr_ft or base_lr / 3, aliases, a.only)
    out_dir = ROOT / "outputs" / "eval" / f"ablation_{a.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res, all_rows = {}, []
    order = [e for e in ALL if e in experiments]
    for exp in order:  # base_en đứng trước ft_vi trong ALL nên checkpoint EN đã sẵn sàng
        spec = experiments[exp]
        ckpt = ROOT / "outputs" / "runs" / spec["run"] / "best.pt"
        print(f"\n===== {exp} (run {spec['run']}) =====", flush=True)
        if "alias_of" in spec:
            res[exp] = dict(res.get(spec["alias_of"], {}))
            continue
        if a.reuse and ckpt.exists():
            print("  dùng lại checkpoint có sẵn")
        else:
            subprocess.run(train_cmd(cfg_path, spec, a.epochs), cwd=ROOT, check=True)
        model, charset, mcfg, _ = load_checkpoint(ckpt, device)
        sets = default_sets(data_root, spec["sets"])
        rows, preds = evaluate_sets(model, charset, mcfg, sets, a.decoders, 32, a.max_samples, device=device)
        save_outputs(out_dir / exp, rows, preds, f"{exp} ({spec['run']})")
        summ = json.loads((ROOT / "outputs" / "runs" / spec["run"] / "summary.json").read_text(encoding="utf-8"))
        res[exp] = dict(rows=rows, summary=summ)
        all_rows += [dict(exp=exp, run=spec["run"], **r) for r in rows]

    if all_rows:
        with open(out_dir / "all_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(all_rows)
    head = (f"# Ablation ({a.mode})\n\nCấu hình: {cfg_path.name}, epochs={a.epochs or 'theo config'}, decoders={a.decoders}\n\n")
    if a.mode == "smoke":
        head += ("> **CẢNH BÁO**: chế độ smoke chỉ kiểm tra pipeline chạy thông (bộ dữ liệu 400 ảnh, vài epoch). "
                 "Số liệu dưới đây KHÔNG cho phép kết luận gì về hiệu quả của từng thành phần.\n\n")
    (out_dir / "ablation_summary.md").write_text(head + build_summary(res, a.decoders), encoding="utf-8")
    print("\n" + (out_dir / "ablation_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
