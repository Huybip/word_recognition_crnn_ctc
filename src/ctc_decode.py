"""Giải mã CTC. Giai đoạn 3 chỉ cần greedy; Giai đoạn 4 bổ sung prefix beam search vào file này."""
from __future__ import annotations

import torch


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor | None = None, blank: int = 0) -> list[list[int]]:
    """Greedy CTC: argmax từng bước -> gộp ký tự lặp liền kề -> bỏ blank.

    log_probs: (T, N, C); lengths: (N,) số bước hợp lệ của từng mẫu (bỏ phần pad).
    Trả về list N danh sách index (không chứa blank).
    """
    best = log_probs.argmax(dim=2).transpose(0, 1).cpu()  # (N, T)
    out = []
    for i in range(best.shape[0]):
        seq = best[i, : int(lengths[i])] if lengths is not None else best[i]
        seq = torch.unique_consecutive(seq).tolist()  # gộp lặp TRƯỚC khi bỏ blank
        out.append([c for c in seq if c != blank])
    return out


# ====================================================================== prefix beam search (Giai đoạn 4)
import math
from collections import defaultdict

import numpy as np

NEG_INF = float("-inf")


def _lse(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) ổn định số học."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def prefix_beam_search(logp, beam_width: int = 10, blank: int = 0, prune: float = -12.0):
    """CTC prefix beam search (Hannun 2014), tính trong không gian log, KHÔNG dùng mô hình ngôn ngữ.

    logp: (T, C) log-xác suất của MỘT mẫu (đã cắt bỏ phần pad).
    Mỗi prefix giữ hai xác suất: p_b (các alignment kết thúc bằng blank) và p_nb (kết thúc bằng ký tự).
      - gặp blank        : prefix giữ nguyên, cộng vào p_b.
      - gặp ký tự c khác ký tự cuối: mở rộng prefix, cộng vào p_nb của prefix mới.
      - gặp ký tự c GIỐNG ký tự cuối: nếu alignment trước đó kết thúc bằng blank (p_b) -> prefix mới (ký tự lặp thật,
        "a-a" -> "aa"); nếu kết thúc bằng ký tự (p_nb) -> gộp vào chính prefix cũ ("aa" -> "a").
    prune: bỏ qua ký tự có log-xác suất < prune ở từng bước (blank luôn được giữ); dùng -inf để tắt (chính xác tuyệt đối khi beam đủ lớn).
    Trả về list [(ids, log_score)] sắp giảm dần theo score; score = log P(prefix | ảnh) đã cộng mọi alignment.
    """
    lp = np.asarray(logp, dtype=np.float64)
    T, C = lp.shape
    beams = {(): (0.0, NEG_INF)}  # prefix -> (log p_b, log p_nb)
    for t in range(T):
        row = lp[t]
        cands = set(np.nonzero(row > prune)[0].tolist())
        cands.add(blank)
        nxt = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (pb, pnb) in beams.items():
            ptot = _lse(pb, pnb)
            last = prefix[-1] if prefix else None
            for c in cands:
                p = row[c]
                if p == NEG_INF:
                    continue
                if c == blank:
                    e = nxt[prefix]
                    e[0] = _lse(e[0], ptot + p)
                elif c == last:
                    e = nxt[prefix + (c,)]          # lặp thật: chỉ đi qua trạng thái kết thúc bằng blank
                    e[1] = _lse(e[1], pb + p)
                    e = nxt[prefix]                 # gộp lặp: ký tự cuối kéo dài thêm một bước
                    e[1] = _lse(e[1], pnb + p)
                else:
                    e = nxt[prefix + (c,)]
                    e[1] = _lse(e[1], ptot + p)
        top = sorted(nxt.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]), reverse=True)[:beam_width]
        beams = {k: (v[0], v[1]) for k, v in top}
    return sorted(((list(k), _lse(*v)) for k, v in beams.items()), key=lambda x: -x[1])


def beam_decode(log_probs: torch.Tensor, lengths: torch.Tensor | None = None, beam_width: int = 10,
                blank: int = 0, prune: float = -12.0) -> list[list[int]]:
    """Prefix beam search cho cả batch. log_probs: (T,N,C); lengths: (N,). Trả list N danh sách index."""
    arr = log_probs.detach().float().cpu().numpy()
    out = []
    for i in range(arr.shape[1]):
        t = int(lengths[i]) if lengths is not None else arr.shape[0]
        out.append(prefix_beam_search(arr[:t, i], beam_width, blank, prune)[0][0])
    return out


def make_pyctcdecode(chars: list[str], beam_width: int = 10):
    """(Tùy chọn) bọc pyctcdecode không dùng LM để đối chiếu. Trả hàm (log_probs, lengths) -> list[list[int]]."""
    from pyctcdecode import build_ctcdecoder
    labels = [""] + list(chars)  # pyctcdecode quy ước nhãn rỗng = blank
    dec = build_ctcdecoder(labels)
    stoi = {c: i + 1 for i, c in enumerate(chars)}

    def run(log_probs, lengths=None):
        arr = log_probs.detach().float().cpu().numpy()
        res = []
        for i in range(arr.shape[1]):
            t = int(lengths[i]) if lengths is not None else arr.shape[0]
            text = dec.decode(arr[:t, i], beam_width=beam_width)
            res.append([stoi[c] for c in text if c in stoi])
        return res
    return run
