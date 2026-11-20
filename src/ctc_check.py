"""Kiểm tra T (số bước thời gian) có đủ cho nhãn hay không.

CTC chỉ căn chỉnh được khi T >= L + R, với L = độ dài nhãn, R = số cặp ký tự liền kề GIỐNG NHAU
(giữa hai ký tự lặp phải có 1 blank, vd "ll" cần ít nhất "l-l" = 3 bước). Đó là điều kiện tối thiểu CHÍNH XÁC.
Ngưỡng bảo thủ trong đề bài 2*L+1 là cận trên của L+R (R <= L-1) và để mô hình có chỗ "thở" giữa các ký tự.

Cảnh báo: nn.CTCLoss(zero_infinity=True) âm thầm đưa loss của mẫu không khả thi về 0, nên phải
kiểm tra TRƯỚC khi train, nếu không mẫu đó bị bỏ qua mà không ai biết.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .dataset import OCRDataset
from .text_utils import Charset


def ctc_min_length(text: str) -> int:
    """Số bước tối thiểu để CTC sinh được `text`: L + số cặp liền kề giống nhau."""
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def collect_stats(split_dirs, charset: Charset, img_height: int, max_width: int | None = 768):
    """Trả list (split, text, width_sau_resize) cho mọi mẫu (chỉ đọc header ảnh)."""
    out = []
    for d in split_dirs:
        ds = OCRDataset(d, charset, img_height=img_height, max_width=max_width)
        name = Path(d).name
        for i, (_, text) in enumerate(ds.samples):
            out.append((name, text, ds.width_of(i)))
    return out


def feasibility_report(stats, strides=(4, 2, 1), conservative_factor: int = 2, img_height: int | None = None):
    """Với mỗi stride s: T = W // s. Đếm mẫu vi phạm điều kiện chính xác (T >= L+R)
    và điều kiện bảo thủ (T >= 2L+1). Trả (text_report, dict số liệu, stride_khuyên_dùng)."""
    lmax = max(len(t) for _, t, _ in stats)
    wmax = max(w for _, _, w in stats)
    wmin = min(w for _, _, w in stats)
    lines = [f"Số mẫu: {len(stats)} | nhãn dài nhất Lmax = {lmax} | rộng min/max sau resize: {wmin}/{wmax} px",
             f"Yêu cầu bảo thủ theo đề bài: T >= 2*Lmax+1 = {2 * lmax + 1}"]
    res, best = {}, None
    for s in strides:
        t_at_max = wmax // s
        bad_exact = [(sp, t, w) for sp, t, w in stats if w // s < ctc_min_length(t)]
        bad_cons = [(sp, t, w) for sp, t, w in stats if w // s < conservative_factor * len(t) + 1]
        res[s] = dict(T_at_max_width=t_at_max, ok_global=t_at_max >= 2 * lmax + 1,
                      n_bad_exact=len(bad_exact), n_bad_conservative=len(bad_cons), examples=bad_exact[:5])
        lines.append(f"  stride {s}: T(rộng max)={t_at_max:>4} "
                     f"{'>=' if t_at_max >= 2 * lmax + 1 else '< '} {2 * lmax + 1} | "
                     f"vi phạm CHÍNH XÁC (T<L+R): {len(bad_exact):>5} | vi phạm bảo thủ (T<2L+1): {len(bad_cons):>5}")
        if best is None and not bad_exact and not bad_cons:
            best = s
    if best is None:  # không stride nào thỏa bảo thủ: chọn stride lớn nhất thỏa điều kiện chính xác
        best = next((s for s in strides if res[s]["n_bad_exact"] == 0), None)
    if best == strides[0]:
        lines.append(f"=> stride {best} ĐỦ cho toàn bộ dữ liệu, giữ nguyên width_strides=[2,2].")
    elif best is not None:
        ws = {2: [2, 1], 1: [1, 1]}[best]
        lines.append(f"=> stride {strides[0]} có mẫu dưới ngưỡng bảo thủ. Hai cách khắc phục:")
        lines.append(f"   (a) giảm stride: cnn.width_strides={ws} (configs/model_s2.yaml), T gấp {strides[0] // best} lần, RNN/CNN chậm hơn;")
        if img_height is None or img_height < 48:
            lines.append("   (b) tăng img_height lên 48 (configs/model_h48.yaml): ảnh giữ tỉ lệ nên rộng ra 1,5 lần, "
                         "đồng thời nét dấu tiếng Việt rõ hơn. Chạy lại với --height 48 để kiểm chứng.")
        else:
            lines.append(f"   (b) img_height đã là {img_height}; chỉ còn vài mẫu font rất hẹp dưới ngưỡng bảo thủ (vẫn thỏa điều kiện chính xác).")
    else:
        lines.append("=> không stride nào đủ: kiểm tra dữ liệu (ảnh quá hẹp so với nhãn) hoặc tăng img_height/giảm max_chars.")
    return "\n".join(lines), res, best


def worst_samples(stats, stride: int, k: int = 5):
    """k mẫu có T/(L+R) nhỏ nhất (rủi ro cao nhất)."""
    return sorted(stats, key=lambda x: (x[2] // stride) / ctc_min_length(x[1]))[:k]
