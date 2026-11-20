"""Test Giai đoạn 2. Chạy: pytest -q tests/test_model.py"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ctc_check import ctc_min_length, feasibility_report  # noqa: E402
from src.model import CRNN, count_params, summary  # noqa: E402

C = 201


def small(**kw):  # mô hình nhỏ cho test nhanh
    d = dict(num_classes=C, channels=(8, 16, 32, 64), rnn_hidden=32)
    d.update(kw)
    return CRNN(**d)


@pytest.mark.parametrize("h,strides,div", [(32, (2, 2), 4), (32, (2, 1), 2), (32, (1, 1), 1), (48, (2, 2), 4), (64, (2, 2), 4)])
def test_output_shape_and_T(h, strides, div):
    m = small(img_height=h, width_strides=strides).eval()
    x = torch.randn(3, 1, h, 200)
    out = m(x)
    assert out.shape == (200 // div, 3, C)  # (T,N,C)
    assert torch.allclose(out.exp().sum(2), torch.ones(200 // div, 3), atol=1e-5)  # log_softmax


@pytest.mark.parametrize("w", [16, 17, 33, 100, 255, 460])
def test_output_lengths_matches_real_T(w):
    m = small().eval()
    T = m(torch.randn(1, 1, 32, w)).shape[0]
    assert int(m.output_lengths(torch.tensor([w]))[0]) == T


def test_invalid_height_raises():
    with pytest.raises(ValueError):
        small(img_height=40)


def test_use_rnn_false_ablation():
    a, b = small(use_rnn=True), small(use_rnn=False)
    assert b.rnn is None and count_params(b)[0] < count_params(a)[0]
    x = torch.randn(2, 1, 32, 128)
    assert b.eval()(x).shape == a.eval()(x).shape == (32, 2, C)


def test_packing_equals_unpacked_when_no_padding():
    torch.manual_seed(0)
    m = small(use_packing=True).eval()
    x, w = torch.randn(3, 1, 32, 96), torch.tensor([96, 96, 96])
    with torch.no_grad():
        a = m(x, w)
        m.use_packing = False
        b = m(x, w)
    assert torch.allclose(a, b, atol=1e-5)


def test_packing_runs_with_padding():
    m = small(use_packing=True).train()
    x, w = torch.randn(3, 1, 32, 128), torch.tensor([128, 64, 40])
    out = m(x, w)
    assert out.shape == (32, 3, C) and torch.isfinite(out).all()


def test_ctc_loss_finite_and_gradients():
    torch.manual_seed(0)
    m = small().train()
    x, w = torch.randn(4, 1, 32, 160), torch.tensor([160, 128, 96, 64])
    tl = torch.tensor([8, 6, 4, 3])
    tg = torch.randint(1, C, (int(tl.sum()),))
    loss = nn.CTCLoss(blank=0, zero_infinity=False)(m(x, w), tg, m.output_lengths(w), tl)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_ctc_infeasible_gives_inf_without_zero_infinity():
    """Chứng minh vì sao phải kiểm tra T: T < L+R thì loss = inf (zero_infinity sẽ che đi)."""
    m = small().eval()
    x, w = torch.randn(1, 1, 32, 32), torch.tensor([32])  # T = 8
    tl, tg = torch.tensor([10]), torch.randint(1, C, (10,))
    out = m(x, w)
    assert torch.isinf(nn.CTCLoss(zero_infinity=False)(out, tg, m.output_lengths(w), tl))
    assert nn.CTCLoss(zero_infinity=True)(out, tg, m.output_lengths(w), tl).item() == 0.0


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = small(rnn_dropout=0.0, head_dropout=0.0).train()
    x, w = torch.randn(4, 1, 32, 192), torch.tensor([192] * 4)
    tl = torch.tensor([5, 4, 6, 3])
    tg = torch.randint(1, 30, (int(tl.sum()),))
    opt, ctc = torch.optim.Adam(m.parameters(), 3e-3), nn.CTCLoss(zero_infinity=True)
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss = ctc(m(x, w), tg, m.output_lengths(w), tl)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first, f"loss không giảm đủ: {first:.2f} -> {loss.item():.2f}"


def test_ctc_min_length():
    assert ctc_min_length("abc") == 3 and ctc_min_length("aab") == 4 and ctc_min_length("lll") == 5 and ctc_min_length("") == 0


def test_feasibility_report_recommends_stride():
    # nhãn 10 ký tự trong ảnh rộng 40 px: stride 4 -> T=10 < 2L+1=21 nhưng >= L (khả thi chính xác);
    stats = [("x", "abcdefghij", 40)] * 3 + [("x", "abc", 400)]
    txt, res, best = feasibility_report(stats)
    assert res[4]["n_bad_exact"] == 0 and res[4]["n_bad_conservative"] == 3
    assert best == 1 and "width_strides=[1, 1]" in txt
    stats2 = [("x", "aaaaaaaaaa", 40)]  # cần L+R = 19 bước; stride 4 -> T=10 vô nghiệm
    assert feasibility_report(stats2)[1][4]["n_bad_exact"] == 1


def test_summary_runs():
    s = summary(small(), width=128)
    assert "(T,N,C)" in s and "Tham số" in s and "LSTM" in s
