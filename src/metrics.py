"""Chỉ số đánh giá: CER (Character Error Rate) và Word Accuracy (khớp hoàn toàn cả dòng).

CER gộp theo toàn tập:  CER = tổng khoảng cách Levenshtein / tổng số ký tự nhãn
(không phải trung bình CER từng mẫu, vì cách đó làm nhãn ngắn có trọng số quá lớn).
Giai đoạn 4 sẽ mở rộng thêm S/D/I và kiểm tra chéo với jiwer.
"""
from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _RF
except Exception:  # không có rapidfuzz -> dùng bản thuần Python
    _RF = None


def levenshtein_py(a: str, b: str) -> int:
    """Khoảng cách Levenshtein (thay/xóa/chèn đều tốn 1), DP hai hàng."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein(a: str, b: str) -> int:
    return _RF.distance(a, b) if _RF is not None else levenshtein_py(a, b)


def cer_stats(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    """(tổng số lỗi chỉnh sửa, tổng số ký tự nhãn) - cộng dồn được qua nhiều batch."""
    assert len(refs) == len(hyps)
    return sum(levenshtein(r, h) for r, h in zip(refs, hyps)), sum(len(r) for r in refs)


def cer(refs: list[str], hyps: list[str]) -> float:
    edits, n = cer_stats(refs, hyps)
    return edits / max(1, n)


def exact_match_count(refs: list[str], hyps: list[str]) -> int:
    return sum(r == h for r, h in zip(refs, hyps))


def word_accuracy(refs: list[str], hyps: list[str]) -> float:
    return exact_match_count(refs, hyps) / max(1, len(refs))


# ====================================================================== S/D/I và căn chỉnh (Giai đoạn 4)
def align(ref: str, hyp: str) -> list[tuple[str, int, int]]:
    """Căn chỉnh Levenshtein tối thiểu. Trả list (op, i, j) với op in {'equal','sub','del','ins'}:
      equal/sub: ref[i] <-> hyp[j] | del: ref[i] bị mất | ins: hyp[j] thừa.
    Hòa điểm ưu tiên: khớp/thay > xóa > chèn. Dùng ở Giai đoạn 5 (ma trận nhầm lẫn)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def edit_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(S, D, I): số ký tự bị thay, bị xóa (có trong nhãn, thiếu trong dự đoán), bị chèn thừa.
    Luôn có S + D + I = levenshtein(ref, hyp)."""
    if _RF is not None:
        tags = [op.tag for op in _RF.editops(ref, hyp)]
        return tags.count("replace"), tags.count("delete"), tags.count("insert")
    ops = [o[0] for o in align(ref, hyp)]
    return ops.count("sub"), ops.count("del"), ops.count("ins")


def corpus_metrics(refs: list[str], hyps: list[str]) -> dict:
    """CER = (S+D+I)/N gộp toàn tập (N = tổng ký tự nhãn) + tách S/D/I + Word Accuracy (khớp hoàn toàn)."""
    assert len(refs) == len(hyps)
    S = D = I = 0
    for r, h in zip(refs, hyps):
        s, d, i = edit_counts(r, h)
        S, D, I = S + s, D + d, I + i
    N = sum(len(r) for r in refs)
    n = max(1, N)
    return dict(n=len(refs), n_chars=N, S=S, D=D, I=I, cer=(S + D + I) / n, sub_rate=S / n, del_rate=D / n,
                ins_rate=I / n, word_acc=exact_match_count(refs, hyps) / max(1, len(refs)))
