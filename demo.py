#!/usr/bin/env python
"""Demo: nhận dạng chữ trong ảnh bằng mô hình CRNN + CTC đã huấn luyện.

Ví dụ:
  python demo.py samples/*.png                                   # greedy, checkpoint mặc định
  python demo.py --ckpt mid_vi --decoder beam10 --verbose anh.png # beam search, in xác suất + n-best
  python demo.py --ckpt outputs/runs/vi_finetune/best.pt thu_muc_anh/
Đầu ra mặc định: <đường dẫn ảnh><TAB><chuỗi dự đoán>.

Lưu ý: mô hình chỉ được huấn luyện trên ảnh SINH (TRDG): chữ tối trên nền sáng, một dòng. Ảnh thật (chụp, nghiêng,
nhiều dòng, nền phức tạp) có thể cho kết quả kém; dùng --invert cho chữ sáng trên nền tối.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.ctc_decode import greedy_decode, prefix_beam_search  # noqa: E402
from src.dataset import preprocess_image  # noqa: E402
from src.eval import load_checkpoint  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@torch.no_grad()
def predict(model, charset, mcfg, image, decoder: str = "greedy", nbest: int = 1, invert: bool = False, device=None):
    """Trả list [(chuỗi, xác suất)] giảm dần theo xác suất.
    greedy: một kết quả; xác suất = xác suất của đường đi argmax (CẬN DƯỚI xác suất của chuỗi vì bỏ qua các alignment khác).
    beamW : n-best từ prefix beam search; xác suất = P(chuỗi | ảnh) đã cộng mọi alignment (xấp xỉ do cắt tỉa)."""
    device = device or next(model.parameters()).device
    x = preprocess_image(image, mcfg["img_height"], invert=invert).to(device)
    lp = model(x)  # (T, 1, C) log-probs
    T = int(model.output_lengths(torch.tensor([x.shape[-1]]))[0])
    lp = lp[:max(1, T)]
    if decoder == "greedy":
        text = charset.decode(greedy_decode(lp)[0])
        return [(text, math.exp(float(lp.max(dim=2).values.sum())))]
    if decoder.startswith("beam"):
        w = max(int(decoder[4:] or 10), nbest)
        res = prefix_beam_search(lp[:, 0].float().cpu().numpy(), beam_width=w)
        return [(charset.decode(ids), math.exp(sc)) for ids, sc in res[:max(1, nbest)]]
    raise ValueError(f"decoder không hỗ trợ: {decoder} (dùng greedy hoặc beam<W>, vd beam10)")


def collect(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="đường dẫn ảnh hoặc thư mục ảnh")
    default = ROOT / "checkpoints" / "mid_vi_best.pt"
    ap.add_argument("--ckpt", default=str(default) if default.exists() else None, help="đường dẫn .pt hoặc tên run trong outputs/runs")
    ap.add_argument("--decoder", default="greedy", help="greedy | beam<W>, vd beam10")
    ap.add_argument("--nbest", type=int, default=1, help="số kết quả hàng đầu (chỉ có nghĩa với beam)")
    ap.add_argument("--invert", action="store_true", help="đảo màu (chữ sáng trên nền tối)")
    ap.add_argument("--verbose", action="store_true", help="in thêm xác suất và n-best")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if not a.ckpt:
        sys.exit("Chưa có checkpoint: truyền --ckpt (đường dẫn .pt hoặc tên run trong outputs/runs).")
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    model, charset, mcfg, ck = load_checkpoint(a.ckpt, device)
    files = collect(a.images)
    if not files:
        sys.exit("Không tìm thấy ảnh nào.")
    n_err = 0
    for f in files:
        try:
            res = predict(model, charset, mcfg, f, a.decoder, a.nbest, a.invert, device)
        except Exception as e:  # ảnh hỏng/không đọc được: báo và đi tiếp
            print(f"{f}\t[LỖI: {e}]", file=sys.stderr)
            n_err += 1
            continue
        print(f"{f}\t{res[0][0]}")
        if a.verbose:
            for i, (t, p) in enumerate(res, 1):
                print(f"    #{i}  p={p:.4f}  {t}")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
