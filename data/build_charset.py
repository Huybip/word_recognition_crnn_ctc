#!/usr/bin/env python
"""Xây bảng ký tự (blank = index 0) từ nhãn đã sinh.

Đầu ra:
  data/charset_en.txt : ký tự của dữ liệu EN
  data/charset.txt    : EN + ký tự chỉ có ở VI (ký tự EN GIỮ NGUYÊN index) -> fine-tune EN->VI
                        chỉ cần mở rộng lớp Linear cuối, copy trọng số các lớp cũ.
  data/charset_info.json
Mặc định "bảo đảm" có đủ a-zA-Z0-9 và toàn bộ 134 chữ cái tiếng Việt có dấu để charset
không phụ thuộc may rủi lúc lấy mẫu (--no_ensure để tắt).
"""
import argparse
import json
import string
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_utils import VI_LETTERS, Charset, nfc  # noqa: E402


def chars_of(lang_dir: Path) -> set:
    cs = set()
    for tsv in lang_dir.glob("*/labels.tsv"):
        for line in tsv.read_text(encoding="utf-8").split("\n"):
            if line:
                cs |= set(nfc(line.split("\t")[1]))
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--root", default=None, help="thư mục chứa en/ và vi/")
    ap.add_argument("--no_ensure", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(a.root or cfg["out_root"])
    root = root if root.is_absolute() else ROOT / root

    en = chars_of(root / "en") if (root / "en").exists() else set()
    vi = chars_of(root / "vi") if (root / "vi").exists() else set()
    if not a.no_ensure:
        en |= set(string.ascii_letters + string.digits)
        vi |= set(string.ascii_letters + string.digits) | set(VI_LETTERS)
    if not en and not vi:
        sys.exit("Không tìm thấy nhãn. Chạy generate_dataset.py trước.")

    en_sorted = sorted(en)
    vi_only = sorted(vi - en)
    full = en_sorted + vi_only
    Charset(en_sorted).save(ROOT / "data" / "charset_en.txt")
    Charset(full).save(ROOT / "data" / "charset.txt")
    info = dict(num_chars_en=len(en_sorted), num_chars_full=len(full), num_classes_full=len(full) + 1,
                blank_index=0, vi_only="".join(vi_only), en_chars="".join(en_sorted))
    (ROOT / "data" / "charset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"charset_en: {len(en_sorted)} ký tự | charset (EN+VI): {len(full)} ký tự -> {len(full) + 1} lớp đầu ra (gồm blank)")
    print(f"Ký tự chỉ có ở VI ({len(vi_only)}): {''.join(vi_only)}")


if __name__ == "__main__":
    main()
