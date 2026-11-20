"""Dataset + collate_fn + BucketBatchSampler cho CRNN/CTC.

Tiền xử lý: grayscale -> resize về chiều cao cố định `img_height` (giữ tỉ lệ) -> chuẩn hóa
[-1, 1]. Batch được pad theo chiều rộng bằng giá trị trung vị của chính ảnh đó (hợp cả nền
sáng lẫn tối). BucketBatchSampler gom các ảnh có chiều rộng gần nhau vào cùng batch để giảm pad.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .text_utils import Charset, nfc


def resized_width(w: int, h: int, height: int, min_width: int = 16, max_width: int | None = 768) -> int:
    """Chiều rộng sau khi resize về cao `height` (giữ tỉ lệ, kẹp trong [min_width, max_width])."""
    new_w = max(min_width, round(w * height / h))
    return min(new_w, max_width) if max_width else new_w


def preprocess_image(img, img_height: int = 32, max_width: int | None = 768, min_width: int = 16,
                     invert: bool = False) -> torch.Tensor:
    """Ảnh (PIL hoặc đường dẫn) -> tensor (1, 1, H, W) đã chuẩn hóa [-1, 1], đúng như lúc huấn luyện.
    Ảnh RGBA được đặt lên nền trắng; invert=True đảo màu cho chữ sáng trên nền tối (mô hình học chữ tối trên nền sáng)."""
    from PIL import ImageOps
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("L")
    if invert:
        img = ImageOps.invert(img)
    w, h = img.size
    img = img.resize((resized_width(w, h, img_height, min_width, max_width), img_height), Image.BICUBIC)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x)[None, None]


class OCRDataset(Dataset):
    def __init__(self, split_dir, charset: Charset, img_height: int = 32,
                 max_width: int | None = 768, min_width: int = 16, cache: bool = False, strict: bool = True):
        self.strict = strict  # False: nhãn có ký tự ngoài charset không gây lỗi (xem oov_rate)
        self.dir = Path(split_dir)
        self.charset, self.H = charset, img_height
        self.max_width, self.min_width = max_width, min_width
        self.samples = []
        for line in (self.dir / "labels.tsv").read_text(encoding="utf-8").split("\n"):
            if line:
                parts = line.split("\t")  # path, text, [font]
                self.samples.append((self.dir / parts[0], nfc(parts[1])))
        self._cache = None
        if cache:
            self.enable_cache()

    def __len__(self):
        return len(self.samples)

    # ---- chọn tập con (dùng cho smoke test / overfit)
    def take(self, n: int, seed: int = 0) -> "OCRDataset":
        """Giữ lại n mẫu ngẫu nhiên (cố định theo seed). Sửa tại chỗ và trả về chính nó."""
        if n and n < len(self.samples):
            keep = sorted(random.Random(seed).sample(range(len(self.samples)), n))
            self.samples = [self.samples[i] for i in keep]
            if self._cache is not None:
                self._cache = [self._cache[i] for i in keep]
        return self

    # ---- kích thước
    def resized_width(self, w: int, h: int) -> int:
        """Chiều rộng sau khi resize về cao H (giữ tỉ lệ, kẹp trong [min_width, max_width])."""
        return resized_width(w, h, self.H, self.min_width, self.max_width)  # ảnh quá dài bị nén ngang (hiếm; xem stats.py)

    def width_of(self, i: int) -> int:
        """Chiều rộng ảnh thứ i sau tiền xử lý, chỉ đọc header (nhanh)."""
        if self._cache is not None:
            return self._cache[i].shape[1]
        w, h = Image.open(self.samples[i][0]).size
        return self.resized_width(w, h)

    def oov_rate(self) -> float:
        """Tỉ lệ mẫu có ít nhất một ký tự nhãn nằm ngoài charset của mô hình."""
        known = self.charset.stoi
        return sum(any(c not in known for c in t) for _, t in self.samples) / max(1, len(self.samples))

    def widths(self) -> list[int]:
        return [self.width_of(i) for i in range(len(self.samples))]

    # ---- đọc ảnh
    def _load_uint8(self, path) -> np.ndarray:
        img = Image.open(path).convert("L")
        w, h = img.size
        img = img.resize((self.resized_width(w, h), self.H), Image.BICUBIC)
        return np.asarray(img, dtype=np.uint8)

    def enable_cache(self):
        """Nạp toàn bộ ảnh (đã resize, uint8) vào RAM: nhanh hơn nhiều khi CPU yếu (Colab)."""
        if self._cache is None:
            self._cache = [self._load_uint8(p) for p, _ in self.samples]

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        x = (arr.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(x).unsqueeze(0)  # (1, H, W)

    def load_image(self, path) -> torch.Tensor:
        return self._to_tensor(self._load_uint8(path))

    def __getitem__(self, i):
        path, text = self.samples[i]
        arr = self._cache[i] if self._cache is not None else self._load_uint8(path)
        return dict(image=self._to_tensor(arr),
                    target=torch.tensor(self.charset.encode(text, self.strict), dtype=torch.long),
                    text=text)


def collate_fn(batch):
    """Trả về dict:
      images         (N, 1, H, Wmax) float32
      targets        (sum L,) long  - nhãn nối phẳng, đúng định dạng nn.CTCLoss
      target_lengths (N,) long
      widths         (N,) long      - chiều rộng thật trước khi pad (để tính input_lengths)
      texts          list[str]
    """
    widths = [b["image"].shape[-1] for b in batch]
    wmax, h = max(widths), batch[0]["image"].shape[1]
    images = torch.empty(len(batch), 1, h, wmax)
    for i, b in enumerate(batch):
        x = b["image"]
        images[i].fill_(x.median().item())
        images[i, :, :, : x.shape[-1]] = x
    return dict(images=images,
                targets=torch.cat([b["target"] for b in batch]),
                target_lengths=torch.tensor([len(b["target"]) for b in batch], dtype=torch.long),
                widths=torch.tensor(widths, dtype=torch.long),
                texts=[b["text"] for b in batch])


class BucketBatchSampler(Sampler):
    """Gom ảnh có chiều rộng gần nhau vào một batch (giảm pad, tăng tốc).

    shuffle=True : xáo trộn, chia thành các "mega-chunk" batch_size*bucket_mult, sắp theo độ rộng
                   trong mỗi chunk, cắt thành batch rồi xáo thứ tự batch (vẫn ngẫu nhiên giữa các epoch).
    shuffle=False: sắp toàn bộ theo độ rộng (dùng cho validation/test).
    """

    def __init__(self, widths, batch_size, shuffle=True, seed=0, bucket_mult=50, drop_last=False):
        self.widths, self.bs, self.shuffle = list(widths), batch_size, shuffle
        self.seed, self.mult, self.drop_last, self.epoch = seed, bucket_mult, drop_last, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        n = len(self.widths)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        idx = list(range(len(self.widths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
            chunk = self.bs * self.mult
            groups = [sorted(idx[i:i + chunk], key=lambda k: self.widths[k]) for i in range(0, len(idx), chunk)]
        else:
            groups = [sorted(idx, key=lambda k: self.widths[k])]
        batches = [g[i:i + self.bs] for g in groups for i in range(0, len(g), self.bs)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.bs]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def make_loader(split_dir, charset, img_height=32, batch_size=64, shuffle=False, num_workers=2, **kw):
    ds = OCRDataset(split_dir, charset, img_height=img_height, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
