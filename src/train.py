#!/usr/bin/env python
"""Huấn luyện CRNN + CTC.

Ví dụ:
  python -m src.train --config configs/small_cpu.yaml                       # smoke test CPU
  python -m src.train --config configs/small_cpu.yaml --overfit 200         # kiểm tra overfit 200 ảnh
  python -m src.train --config configs/full_gpu.yaml                        # pha (a): train EN
  python -m src.train --config configs/full_gpu_vi_finetune.yaml            # pha (b): fine-tune EN -> VI
  python -m src.train --config configs/full_gpu_vi_scratch.yaml             # đối chứng: VI từ đầu
Ghi đè cấu hình:  --set optim.lr=0.0005 train.epochs=5     Tiếp tục: --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .ctc_check import ctc_min_length
from .ctc_decode import greedy_decode
from .dataset import BucketBatchSampler, OCRDataset, collate_fn
from .metrics import cer_stats, exact_match_count
from .model import build_model, count_params, expand_fc
from .text_utils import Charset

ROOT = Path(__file__).resolve().parents[1]


def P(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ------------------------------------------------------------------ cấu hình
def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = deep_update(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)  # kế thừa: base: full_gpu.yaml
    return deep_update(_load_yaml(path.parent / base), cfg) if base else cfg


def _parse_value(v: str):
    """yaml.safe_load nhưng hiểu cả '3e-4' (YAML 1.1 chỉ nhận '3.0e-4' và trả chuỗi nếu thiếu dấu chấm)."""
    x = yaml.safe_load(v)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            pass
    return x


def load_config(path, overrides=()) -> dict:
    cfg = _load_yaml(P(path))
    for ov in overrides:
        key, val = ov.split("=", 1)
        d, parts = cfg, key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _parse_value(val)  # ép kiểu: 1e-3, null, true, [2,1]...
    for k in ("lr", "weight_decay", "clip_grad_norm"):  # file YAML viết "1e-3" cũng thành chuỗi -> ép về float
        if isinstance(cfg.get("optim", {}).get(k), str):
            cfg["optim"][k] = float(cfg["optim"][k])
    return cfg


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


# ------------------------------------------------------------------ dữ liệu
def build_data(cfg, charset):
    d, H, seed = cfg["data"], cfg["model"]["img_height"], cfg["run"]["seed"]
    root = P(d["root"]) / d["lang"]
    kw = dict(img_height=H, max_width=d.get("max_width", 768))
    train_ds = OCRDataset(root / d["train_split"], charset, **kw)
    n_over = cfg.get("debug", {}).get("overfit_n", 0)
    if n_over:  # kiểm tra overfit: train và val trên CÙNG n mẫu
        train_ds.take(n_over, seed)
        val_ds = train_ds
    else:
        if d.get("max_train_samples"):
            train_ds.take(d["max_train_samples"], seed)
        val_ds = OCRDataset(root / d["val_split"], charset, **kw)
        if d.get("max_val_samples"):
            val_ds.take(d["max_val_samples"], seed)
    if d.get("cache_in_ram", False):  # nạp ảnh vào RAM SAU khi đã chọn tập con
        train_ds.enable_cache()
        val_ds.enable_cache()
    bs, nw = d["batch_size"], d.get("num_workers", 0)
    t_w, v_w = train_ds.widths(), val_ds.widths()
    if d.get("bucket", True):
        t_sampler = BucketBatchSampler(t_w, bs, shuffle=True, seed=seed)
        v_sampler = BucketBatchSampler(v_w, bs, shuffle=False)
        mk = lambda ds, s: DataLoader(ds, batch_sampler=s, num_workers=nw, collate_fn=collate_fn,  # noqa: E731
                                      pin_memory=torch.cuda.is_available())
        return train_ds, val_ds, mk(train_ds, t_sampler), mk(val_ds, v_sampler), t_sampler, t_w
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_fn, generator=g,
                    pin_memory=torch.cuda.is_available())
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn)
    return train_ds, val_ds, tl, vl, None, t_w


# ------------------------------------------------------------------ mô hình, khởi tạo từ checkpoint
def load_pretrained(model, ckpt_path, charset: Charset) -> str:
    ck = torch.load(P(ckpt_path), map_location="cpu", weights_only=True)
    old_chars, sd = ck["charset"], ck["model"]
    if charset.chars[: len(old_chars)] != old_chars:
        raise ValueError("Charset mới phải BẮT ĐẦU bằng charset cũ (cùng thứ tự) thì mới giữ được trọng số lớp cuối. "
                         "Hãy dùng data/charset_en.txt cho pha EN và data/charset.txt cho pha VI.")
    own = model.state_dict()
    bad = [k for k, v in sd.items() if not k.startswith("fc.") and (k not in own or own[k].shape != v.shape)]
    if bad:
        raise ValueError(f"Kiến trúc không khớp checkpoint (img_height/channels/hidden?): {bad[:3]}")
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    rep = expand_fc(model, sd["fc.weight"], sd["fc.bias"])
    return (f"Nạp {ckpt_path}: giữ {rep['n_old']} lớp đầu ra cũ (gồm blank), thêm {rep['n_added']} lớp mới "
            f"-> {rep['n_new']} lớp")


def build_optimizer(model, o):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]  # bias, BatchNorm: không weight decay
    return torch.optim.AdamW([dict(params=decay, weight_decay=o["weight_decay"]),
                              dict(params=no_decay, weight_decay=0.0)],
                             lr=o["lr"], betas=tuple(o.get("betas", (0.9, 0.999))))


def build_scheduler(opt, s, lr, total_steps):
    if s["name"] == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total_steps, pct_start=s.get("pct_start", 0.15),
            anneal_strategy="cos", div_factor=s.get("div_factor", 25), final_div_factor=s.get("final_div_factor", 1000))
    if s["name"] == "cosine":  # warmup tuyến tính rồi cosine về lr * min_lr_ratio
        warm, mn = int(s.get("warmup_steps", 0)), s.get("min_lr_ratio", 0.01)

        def f(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total_steps - warm)
            return mn + (1 - mn) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)
    raise ValueError(f"scheduler không hỗ trợ: {s['name']}")


# ------------------------------------------------------------------ đánh giá
@torch.no_grad()
def evaluate(model, loader, ctc, charset, device, use_amp):
    model.eval()
    tot_loss, n, refs, hyps = 0.0, 0, [], []
    for b in loader:
        x, w = b["images"].to(device), b["widths"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x, w)
        il = model.output_lengths(w).clamp(max=out.shape[0])
        loss = ctc(out, b["targets"], il, b["target_lengths"])
        tot_loss += loss.item() * len(w)
        n += len(w)
        refs += b["texts"]
        hyps += [charset.decode(ids) for ids in greedy_decode(out, il)]
    edits, chars = cer_stats(refs, hyps)
    return dict(loss=tot_loss / max(1, n), cer=edits / max(1, chars),
                word_acc=exact_match_count(refs, hyps) / max(1, n), n=n), refs, hyps


# ------------------------------------------------------------------ chọn checkpoint tốt nhất
def is_improved(val: dict, best: dict, min_delta: float = 0.0, plateau: float = 0.8) -> bool:
    """Epoch hiện tại có tốt hơn `best` không?

    Ở giai đoạn "cao nguyên" của CTC (CER >= 80%: mô hình chưa học được gì đáng kể, thường chỉ dự đoán blank) CER dao động ngẫu nhiên
    và không phản ánh tiến bộ: một epoch xuất vài ký tự bậy có thể cho CER 91% dù loss tệ hơn nhiều, rồi chặn
    mọi epoch sau và làm early stopping dừng oan. Vì vậy CER >= plateau được coi như 1.0 và so sánh bằng val loss.
    """
    if best["epoch"] == 0:
        return True
    c, b = (1.0 if val["cer"] >= plateau else val["cer"]), (1.0 if best["cer"] >= plateau else best["cer"])
    return c < b - min_delta or (abs(c - b) < 1e-12 and val["loss"] < best["loss"])


# ------------------------------------------------------------------ chính
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè: khóa.con=giá_trị")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="train+val trên N mẫu để kiểm tra mô hình có học được không")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.overfit:
        cfg.setdefault("debug", {})["overfit_n"] = a.overfit
    run, tr = cfg["run"], cfg["train"]
    seed_everything(run["seed"], run.get("deterministic", False))
    dev_name = run.get("device", "auto")
    device = torch.device("cuda" if dev_name == "auto" and torch.cuda.is_available() else ("cpu" if dev_name == "auto" else dev_name))
    use_amp = bool(tr.get("amp", False)) and device.type == "cuda"
    if device.type == "cpu":
        torch.set_num_threads(max(1, run.get("cpu_threads", torch.get_num_threads())))

    out_dir = P(run["out_dir"]) / run["name"]
    if out_dir.exists() and not (a.resume or a.overwrite):
        sys.exit(f"{out_dir} đã tồn tại. Dùng --resume để tiếp tục hoặc --overwrite để ghi đè.")
    if a.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    charset = Charset.load(P(cfg["data"]["charset"]))
    model = build_model(charset.num_classes, cfg["model"]).to(device)
    print(f"Thiết bị: {device} (AMP={use_amp}) | charset {cfg['data']['charset']}: {len(charset)} ký tự -> {charset.num_classes} lớp")
    print(f"Mô hình: {count_params(model)[0]:,} tham số | H={cfg['model']['img_height']} | stride W = {model.total_stride}")
    if cfg.get("init_from") and not a.resume:
        print(load_pretrained(model, cfg["init_from"], charset))

    t0 = time.time()
    train_ds, val_ds, train_loader, val_loader, sampler, t_widths = build_data(cfg, charset)
    print(f"Dữ liệu {cfg['data']['lang']}: train {len(train_ds)} | val {len(val_ds)} | {len(train_loader)} batch/epoch "
          f"(đọc kích thước: {time.time() - t0:.1f}s)")

    # kiểm tra khả thi CTC trước khi train (zero_infinity sẽ che mẫu vô nghiệm)
    s = model.total_stride
    n_bad = sum(w // s < ctc_min_length(t) for w, (_, t) in zip(t_widths, train_ds.samples))
    print(f"Mẫu train vô nghiệm với CTC (T < L+R): {n_bad}/{len(train_ds)}")
    if n_bad / len(train_ds) > tr.get("max_infeasible_frac", 0.01):
        sys.exit("Quá nhiều mẫu vô nghiệm với CTC: giảm stride hoặc tăng img_height (xem python -m src.check_model).")

    opt = build_optimizer(model, cfg["optim"])
    epochs, steps_per_epoch = tr["epochs"], len(train_loader)
    sched = build_scheduler(opt, cfg["sched"], cfg["optim"]["lr"], epochs * steps_per_epoch)
    ScalerCls = getattr(torch.amp, "GradScaler", None)
    scaler = ScalerCls("cuda", enabled=use_amp) if ScalerCls else torch.cuda.amp.GradScaler(enabled=use_amp)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    es = tr.get("early_stop", {})
    patience = 10**9 if cfg.get("debug", {}).get("overfit_n") else es.get("patience", 10**9)

    start_epoch, best, bad_epochs = 1, dict(cer=float("inf"), loss=float("inf"), epoch=0), 0
    if a.resume:
        ck = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad_epochs = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
        print(f"Tiếp tục từ epoch {start_epoch} (best CER {best['cer']:.4f} @ epoch {best['epoch']})")

    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy(P(cfg["data"]["charset"]), out_dir / "charset.txt")
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        print("(không có tensorboard: chỉ ghi CSV)")
    csv_path = out_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_cer", "val_word_acc", "infeasible_frac", "skipped_steps", "grad_norm", "sec"]
    if not (a.resume and csv_path.exists()):
        csv_path.write_text(",".join(fields) + "\n")

    first_val = None
    for epoch in range(start_epoch, epochs + 1):
        te = time.time()
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        tot, cnt, n_inf, n_samp, skipped, gns = 0.0, 0, 0, 0, 0, []
        for step, b in enumerate(train_loader, 1):
            x, w = b["images"].to(device, non_blocking=True), b["widths"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x, w)
            il = model.output_lengths(w).clamp(max=out.shape[0])
            n_inf += sum(int(i) < ctc_min_length(t) for i, t in zip(il.tolist(), b["texts"]))
            n_samp += len(w)
            loss = ctc(out.float(), b["targets"], il, b["target_lengths"])
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):  # NaN (vd: fp16 tràn số) -> bỏ batch
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("clip_grad_norm", 5.0))
            if torch.isfinite(gn):
                scaler.step(opt)
                gns.append(float(gn))
            else:
                skipped += 1
            scaler.update()
            sched.step()
            tot += loss.item() * len(w)
            cnt += len(w)
            if tr.get("log_every") and step % tr["log_every"] == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} loss {tot / max(1, cnt):.4f}", flush=True)
        val, refs, hyps = evaluate(model, val_loader, ctc, charset, device, use_amp)
        first_val = first_val or val
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tot / max(1, cnt), val_loss=val["loss"],
                   val_cer=val["cer"], val_word_acc=val["word_acc"], infeasible_frac=n_inf / max(1, n_samp),
                   skipped_steps=skipped, grad_norm=float(np.mean(gns)) if gns else float("nan"), sec=time.time() - te)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow([f"{row[k]:.6g}" if isinstance(row[k], float) else row[k] for k in fields])
        if writer:
            for k in fields[1:-1]:
                writer.add_scalar(k.replace("_", "/", 1), row[k], epoch)
        improved = is_improved(val, best, es.get("min_delta", 0.0))
        if improved:
            best, bad_epochs = dict(cer=val["cer"], loss=val["loss"], epoch=epoch), 0
            torch.save(dict(model=model.state_dict(), charset=charset.chars, model_cfg=cfg["model"], epoch=epoch,
                            val=val), out_dir / "best.pt")
            with open(out_dir / "best_val_predictions.tsv", "w", encoding="utf-8") as f:
                f.write("ref\thyp\n" + "\n".join(f"{r}\t{h}" for r, h in zip(refs, hyps)) + "\n")
        else:
            bad_epochs += 1
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best=best, bad_epochs=bad_epochs,
                        charset=charset.chars, model_cfg=cfg["model"]), out_dir / "last.pt")
        print(f"ep {epoch:3d}/{epochs} | lr {row['lr']:.2e} | train {row['train_loss']:.3f} | val {val['loss']:.3f} | "
              f"CER {100 * val['cer']:6.2f}% | WordAcc {100 * val['word_acc']:5.1f}% | vô nghiệm {100 * row['infeasible_frac']:.2f}% | "
              f"bỏ {skipped} | gnorm {row['grad_norm']:.2f} | {row['sec']:.1f}s{' *' if improved else ''}", flush=True)
        if bad_epochs >= patience:
            print(f"Early stopping: {patience} epoch không cải thiện CER.")
            break

    if writer:
        writer.close()
    summary = dict(run=run["name"], device=str(device), amp=use_amp, torch=torch.__version__,
                   params=count_params(model)[0], best_epoch=best["epoch"], best_val_cer=best["cer"], best_val_loss=best["loss"],
                   first_epoch_val=first_val, last_epoch_val=val, epochs_run=epoch, lang=cfg["data"]["lang"],
                   overfit_n=cfg.get("debug", {}).get("overfit_n", 0))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Xong. Tốt nhất: epoch {best['epoch']} | val CER {100 * best['cer']:.2f}% | checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
