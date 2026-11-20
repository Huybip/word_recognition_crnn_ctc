"""CRNN = CNN (trích đặc trưng) + BiLSTM (mô hình chuỗi) + Linear, đầu ra cho CTC.

Luồng shape (img_height=32, width_strides=(2,2)):
  (N,1,32,W) -> conv/pool -> (N,512,1,T) với T = W // 4 -> (T,N,512)
  -> BiLSTM x2 -> (T,N,512) -> Linear -> (T,N,C) -> log_softmax
Lớp CNN chỉ giảm chiều cao tới 1; chiều rộng được giữ lại làm trục thời gian T.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 32, in_channels: int = 1,
                 channels=(64, 128, 256, 512), width_strides=(2, 2),
                 use_rnn: bool = True, rnn_hidden: int = 256, rnn_layers: int = 2,
                 rnn_dropout: float = 0.2, use_packing: bool = False, head_dropout: float = 0.1):
        super().__init__()
        if img_height % 16 != 0 or img_height < 16:
            raise ValueError(f"img_height phải chia hết cho 16 (nhận {img_height})")
        if any(s not in (1, 2) for s in width_strides):
            raise ValueError("width_strides chỉ nhận 1 hoặc 2")
        c1, c2, c3, c4 = channels
        sw1, sw2 = width_strides
        self.num_classes, self.img_height = num_classes, img_height
        self.use_rnn, self.use_packing = use_rnn, use_packing
        self.total_stride = sw1 * sw2  # tỉ lệ giảm chiều rộng: T = W // total_stride

        self.cnn = nn.Sequential(
            _conv_bn_relu(in_channels, c1), nn.MaxPool2d((2, sw1)),   # H/2
            _conv_bn_relu(c1, c2), nn.MaxPool2d((2, sw2)),            # H/4
            _conv_bn_relu(c2, c3), _conv_bn_relu(c3, c3), nn.MaxPool2d((2, 1)),  # H/8
            _conv_bn_relu(c3, c4), _conv_bn_relu(c4, c4), nn.MaxPool2d((2, 1)),  # H/16
            # kernel cao = H/16 (không padding) gom chiều cao còn lại về 1, giữ nguyên W
            nn.Conv2d(c4, c4, (img_height // 16, 1), bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(head_dropout)
        if use_rnn:
            self.rnn = nn.LSTM(c4, rnn_hidden, num_layers=rnn_layers, bidirectional=True,
                               dropout=rnn_dropout if rnn_layers > 1 else 0.0)
            feat = 2 * rnn_hidden
        else:
            self.rnn, feat = None, c4
        self.fc = nn.Linear(feat, num_classes)

    # ---- độ dài chuỗi thời gian
    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """Số bước thời gian T của từng ảnh thật (trước pad) -> dùng làm input_lengths cho CTCLoss."""
        return torch.as_tensor(widths, dtype=torch.long) // self.total_stride

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N,1,H,W) -> log-probs (T,N,C). `widths` chỉ cần khi use_packing=True."""
        f = self.cnn(x)
        if f.shape[2] != 1:
            raise RuntimeError(f"CNN phải thu chiều cao về 1, nhận {tuple(f.shape)}; kiểm tra img_height={self.img_height}")
        f = self.drop(f.squeeze(2).permute(2, 0, 1))  # (T,N,C)
        if self.rnn is not None:
            if self.use_packing and widths is not None:
                lens = self.output_lengths(widths).clamp(min=1).cpu()
                packed = pack_padded_sequence(f, lens, enforce_sorted=False)
                f, _ = pad_packed_sequence(self.rnn(packed)[0], total_length=f.shape[0])
            else:
                f, _ = self.rnn(f)
        return F.log_softmax(self.fc(f).float(), dim=2)  # log_softmax luôn ở float32 cho CTC


def build_model(num_classes: int, cfg: dict) -> CRNN:
    r = cfg["rnn"]
    return CRNN(num_classes, img_height=cfg["img_height"], channels=tuple(cfg["cnn"]["channels"]),
                width_strides=tuple(cfg["cnn"]["width_strides"]), use_rnn=r["use_rnn"],
                rnn_hidden=r["hidden"], rnn_layers=r["layers"], rnn_dropout=r["dropout"],
                use_packing=r.get("use_packing", False), head_dropout=cfg.get("head_dropout", 0.1))


def count_params(m: nn.Module) -> tuple[int, int]:
    tot = sum(p.numel() for p in m.parameters())
    return tot, sum(p.numel() for p in m.parameters() if p.requires_grad)


def summary(model: CRNN, width: int = 320, batch: int = 2) -> str:
    """In shape tensor sau từng tầng Conv/Pool/LSTM/Linear và số tham số."""
    rows, hooks = [], []
    kinds = (nn.Conv2d, nn.MaxPool2d, nn.LSTM, nn.Linear)

    def make(name):
        def hook(m, _inp, out):
            o = out[0] if isinstance(m, nn.LSTM) else out
            rows.append((name, type(m).__name__, tuple(o.shape), sum(p.numel() for p in m.parameters())))
        return hook

    for name, m in model.named_modules():
        if isinstance(m, kinds):
            hooks.append(m.register_forward_hook(make(name)))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    x = torch.zeros(batch, 1, model.img_height, width, device=dev)
    with torch.no_grad():
        out = model(x)
    for h in hooks:
        h.remove()
    model.train(was_training)

    L = [f"Input: {tuple(x.shape)} (N,1,H,W)", f"{'tầng':<14}{'loại':<11}{'shape đầu ra':<24}{'#tham số':>10}", "-" * 59]
    for name, kind, shape, n in rows:
        L.append(f"{name:<14}{kind:<11}{str(shape):<24}{n:>10,}")
        if name == f"cnn.{len(model.cnn) - 3}":  # lớp conv cuối của CNN
            L.append(f"  -> sau BN+ReLU, squeeze chiều cao, permute: (T,N,C)=({shape[3]},{batch},{shape[1]})")
    L.append("-" * 59)
    tot, train = count_params(model)
    n_cnn = sum(p.numel() for p in model.cnn.parameters())
    n_rnn = sum(p.numel() for p in model.rnn.parameters()) if model.rnn is not None else 0
    n_fc = sum(p.numel() for p in model.fc.parameters())
    L.append(f"Output log_softmax: {tuple(out.shape)} (T,N,C) | T = W//{model.total_stride} = {width}//{model.total_stride} = {out.shape[0]}")
    L.append(f"Tham số: tổng {tot:,} (huấn luyện {train:,}) | CNN {n_cnn:,} | RNN {n_rnn:,} | FC {n_fc:,}")
    return "\n".join(L)


def expand_fc(model: CRNN, old_weight: torch.Tensor, old_bias: torch.Tensor) -> dict:
    """Mở rộng lớp Linear cuối từ n_old lớp lên model.num_classes lớp, GIỮ trọng số cũ.

    Điều kiện: charset mới bắt đầu bằng charset cũ (build_charset.py bảo đảm), nên blank (0)
    và mọi ký tự cũ giữ nguyên index -> hàng trọng số cũ copy y nguyên vào đúng vị trí.
    Hàng mới (ký tự mới) khởi tạo N(0, std của trọng số cũ), bias = trung bình bias cũ,
    để các lớp mới không lấn át lớp cũ lúc đầu nhưng vẫn học được nhanh.
    """
    n_old, n_new = old_weight.shape[0], model.fc.out_features
    if model.fc.in_features != old_weight.shape[1]:
        raise ValueError(f"Kích thước đặc trưng khác nhau: {old_weight.shape[1]} vs {model.fc.in_features}")
    if n_new < n_old:
        raise ValueError(f"Charset mới ({n_new} lớp) nhỏ hơn charset cũ ({n_old} lớp)")
    with torch.no_grad():
        model.fc.weight[:n_old] = old_weight
        model.fc.bias[:n_old] = old_bias
        if n_new > n_old:
            model.fc.weight[n_old:].normal_(0.0, float(old_weight.std()))
            model.fc.bias[n_old:] = old_bias.mean()
    return dict(n_old=n_old, n_new=n_new, n_added=n_new - n_old)
