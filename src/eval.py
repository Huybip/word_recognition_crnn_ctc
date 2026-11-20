#!/usr/bin/env python
"""Đánh giá một checkpoint trên các tập test, mỗi tập x mỗi decoder.

Ví dụ:
  python -m src.eval --ckpt en_base                                  # mọi tập {en,vi}/{test,hard}, greedy + beam 5/10/20
  python -m src.eval --ckpt vi_finetune --sets vi/test vi/hard --decoders greedy beam10
  python -m src.eval --ckpt outputs/runs/vi_finetune/best.pt --max_samples 500 --tag quick

Decoder: greedy | beam<W> (prefix beam search tự cài) | pyctc<W> (pyctcdecode, tùy chọn).
Đầu ra (outputs/eval/<tag>/): results.csv, results.md, results.json, preds_<tập>_<decoder>.tsv.

Cách đo thời gian: một lần forward cho mỗi batch rồi chạy lần lượt từng decoder trên CÙNG log-probs.
  ms_model = forward mô hình / ảnh; ms_decode = riêng decoder / ảnh; ms/ảnh = tổng hai phần.
  Không tính thời gian đọc ảnh; khởi động nóng 1 batch trước khi đo; trên GPU có đồng bộ trước khi bấm giờ.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .ctc_decode import beam_decode, greedy_decode, make_pyctcdecode
from .dataset import BucketBatchSampler, OCRDataset, collate_fn
from .metrics import corpus_metrics, levenshtein
from .model import build_model
from .text_utils import Charset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECODERS = ["greedy", "beam5", "beam10", "beam20"]


def resolve_ckpt(x: str) -> Path:
    """'en_base' -> outputs/runs/en_base/best.pt; hoặc đường dẫn .pt."""
    p = Path(x)
    if p.suffix == ".pt":
        return p if p.is_absolute() else ROOT / p
    return ROOT / "outputs" / "runs" / x / "best.pt"


def load_checkpoint(path, device):
    ck = torch.load(resolve_ckpt(str(path)), map_location="cpu", weights_only=True)
    charset = Charset(ck["charset"])
    model = build_model(charset.num_classes, ck["model_cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), charset, ck["model_cfg"], ck


def make_decoder(name: str, charset: Charset):
    if name == "greedy":
        return lambda lp, ln: greedy_decode(lp, ln)
    if name.startswith("beam"):
        w = int(name[4:])
        return lambda lp, ln: beam_decode(lp, ln, beam_width=w)
    if name.startswith("pyctc"):
        return make_pyctcdecode(charset.chars, int(name[5:] or 10))
    raise ValueError(f"decoder không hỗ trợ: {name}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_sets(model, charset, mcfg, sets: dict, decoders, batch_size=32, max_samples=None, seed=0, device=None):
    """sets: {tên: thư mục tập}. Trả (rows, preds) với preds[(tên_tập, decoder)] = list (path, ref, hyp)."""
    device = device or next(model.parameters()).device
    decs = {d: make_decoder(d, charset) for d in decoders}
    rows, preds = [], {}
    for sname, sdir in sets.items():
        ds = OCRDataset(sdir, charset, img_height=mcfg["img_height"], strict=False)
        if max_samples:
            ds.take(max_samples, seed)
        sampler = BucketBatchSampler(ds.widths(), batch_size, shuffle=False)
        hyps = {d: [] for d in decs}
        refs, paths = [], []
        t_model, t_dec = 0.0, {d: 0.0 for d in decs}
        first = True
        for idxs in sampler:
            b = collate_fn([ds[i] for i in idxs])
            x, w = b["images"].to(device), b["widths"]
            if first:  # khởi động nóng, không tính giờ
                model(x, w)
                first = False
            _sync(device)
            t0 = time.perf_counter()
            out = model(x, w)
            _sync(device)
            t_model += time.perf_counter() - t0
            il = model.output_lengths(w).clamp(max=out.shape[0])
            refs += b["texts"]
            paths += [str(ds.samples[i][0].relative_to(ds.dir)) for i in idxs]
            for d, fn in decs.items():
                t0 = time.perf_counter()
                ids = fn(out, il)
                t_dec[d] += time.perf_counter() - t0
                hyps[d] += [charset.decode(i) for i in ids]
        n = len(refs)
        for d in decs:
            m = corpus_metrics(refs, hyps[d])
            rows.append(dict(set=sname, decoder=d, n=n, oov_rate=ds.oov_rate(), cer=m["cer"], sub=m["sub_rate"],
                             dele=m["del_rate"], ins=m["ins_rate"], word_acc=m["word_acc"],
                             ms_model=1000 * t_model / n, ms_decode=1000 * t_dec[d] / n,
                             ms_total=1000 * (t_model + t_dec[d]) / n))
            preds[(sname, d)] = list(zip(paths, refs, hyps[d]))
    return rows, preds


def to_markdown(rows, title="") -> str:
    L = [f"### {title}" if title else "", "",
         "| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |", "|---|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['set']} | {r['decoder']} | {100 * r['cer']:.2f} | {100 * r['word_acc']:.2f} | {r['ms_total']:.2f} |")
    L += ["", "Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.", "",
          "| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['set']} | {r['decoder']} | {r['n']} | {100 * r['sub']:.2f} | {100 * r['dele']:.2f} | {100 * r['ins']:.2f} | "
                 f"{100 * r['oov_rate']:.1f} | {r['ms_model']:.2f} | {r['ms_decode']:.2f} |")
    return "\n".join(L) + "\n"


def save_outputs(out_dir: Path, rows, preds, title=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(to_markdown(rows, title), encoding="utf-8")
    for (sname, d), items in preds.items():
        with open(out_dir / f"preds_{sname.replace('/', '_')}_{d}.tsv", "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\tedit_distance\n")
            for p, r, h in items:
                f.write(f"{p}\t{r}\t{h}\t{levenshtein(r, h)}\n")


def default_sets(root: Path, sets=None):
    names = sets or [f"{l}/{s}" for l in ("en", "vi") for s in ("test", "hard")]
    return {n.replace("/", "_"): root / n for n in names if (root / n / "labels.tsv").exists()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="tên run (en_base) hoặc đường dẫn .pt")
    ap.add_argument("--sets", nargs="*", default=None, help="vd: en/test vi/test en/hard vi/hard (mặc định: tất cả có sẵn)")
    ap.add_argument("--decoders", nargs="*", default=DEFAULT_DECODERS)
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    model, charset, mcfg, ck = load_checkpoint(a.ckpt, device)
    root = Path(a.root) if Path(a.root).is_absolute() else ROOT / a.root
    sets = default_sets(root, a.sets)
    if not sets:
        raise SystemExit("Không tìm thấy tập dữ liệu nào để đánh giá.")
    tag = a.tag or (Path(a.ckpt).parent.name if a.ckpt.endswith(".pt") else a.ckpt)
    print(f"Checkpoint {a.ckpt}: epoch {ck['epoch']}, {charset.num_classes} lớp, H={mcfg['img_height']}, thiết bị {device}")
    rows, preds = evaluate_sets(model, charset, mcfg, sets, a.decoders, a.batch_size, a.max_samples, device=device)
    out_dir = ROOT / "outputs" / "eval" / tag
    save_outputs(out_dir, rows, preds, title=f"Kết quả: {tag}")
    print((out_dir / "results.md").read_text(encoding="utf-8"))
    print(f"Đã lưu vào {out_dir}")


if __name__ == "__main__":
    main()
