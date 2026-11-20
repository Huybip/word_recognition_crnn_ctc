"""Tiện ích văn bản: chuẩn hóa NFC, bỏ dấu, bảng ký tự (Charset) cho CTC.

Quy ước Charset: index 0 = blank (CTC); ký tự thật bắt đầu từ index 1.
File charset.txt: mỗi dòng một token; dòng 0 là `<blank>`; dấu cách ghi là `<space>`.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

BLANK = "<blank>"
SPACE = "<space>"

# 67 chữ thường có dấu/biến thể của tiếng Việt (dạng NFC dựng sẵn)
VI_LOWER = unicodedata.normalize(
    "NFC",
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ",
)
VI_UPPER = unicodedata.normalize("NFC", VI_LOWER.upper())
VI_LETTERS = VI_LOWER + VI_UPPER


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_diacritics(s: str) -> str:
    """Bỏ toàn bộ dấu (kể cả đ->d). Dùng cho CER 'không dấu' ở giai đoạn 5."""
    s = s.replace("đ", "d").replace("Đ", "D")
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


class Charset:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("Charset có ký tự trùng lặp")
        if any(len(c) != 1 for c in chars):
            raise ValueError("Mỗi phần tử Charset phải là đúng 1 ký tự")
        self.chars = list(chars)
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 dành cho blank

    @property
    def num_classes(self) -> int:  # số lớp đầu ra của mô hình (gồm blank)
        return len(self.chars) + 1

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str, strict: bool = True) -> list[int]:
        """strict=False: bỏ qua ký tự ngoài charset (chỉ dùng khi đánh giá model EN trên nhãn VI)."""
        text = nfc(text)
        if not strict:
            return [self.stoi[c] for c in text if c in self.stoi]
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise KeyError(f"Ký tự {e.args[0]!r} (U+{ord(e.args[0]):04X}) không có trong charset") from None

    def decode(self, ids) -> str:
        """Ánh xạ index -> ký tự, bỏ blank. KHÔNG gộp lặp (việc đó thuộc về CTC decoder)."""
        return "".join(self.chars[i - 1] for i in ids if i != 0)

    def save(self, path) -> None:
        lines = [BLANK] + [SPACE if c == " " else c for c in self.chars]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Charset":
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not lines or lines[0] != BLANK:
            raise ValueError("Dòng đầu của charset phải là <blank> (index 0)")
        return cls([" " if t == SPACE else t for t in lines[1:]])
