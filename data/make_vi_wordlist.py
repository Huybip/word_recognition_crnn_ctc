#!/usr/bin/env python
"""Tạo data/vi_words.txt (từ/cụm từ tiếng Việt có dấu, NFC, mỗi dòng một mục).

LƯU Ý: file dicts/vi.txt đi kèm TRDG KHÔNG phải tiếng Việt có dấu (không có ă, ơ, ư, đ...),
nên không dùng. Mặc định tải Viet74K (duyet/vietnamese-wordlist) rồi làm sạch.
Dùng nguồn khác: --input my_words.txt (mỗi dòng một từ/cụm từ).
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_utils import VI_LOWER, nfc  # noqa: E402

URL = "https://raw.githubusercontent.com/duyet/vietnamese-wordlist/master/Viet74K.txt"
# chỉ chữ thường tiếng Việt + dấu cách; không có f, j, w, z (không thuộc tiếng Việt)
TOKEN = f"[a-eg-ik-vxy{VI_LOWER}]+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--input", default=None, help="file local thay vì tải về")
    ap.add_argument("--out", default=str(ROOT / "data" / "vi_words.txt"))
    ap.add_argument("--max_syllables", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=25)
    a = ap.parse_args()

    if a.input:
        raw = Path(a.input).read_text(encoding="utf-8")
    else:
        print(f"Tải {a.url}")
        raw = urllib.request.urlopen(a.url, timeout=60).read().decode("utf-8")

    pat = re.compile(rf"^{TOKEN}( {TOKEN})*$")
    kept, seen = [], set()
    for line in raw.splitlines():
        w = re.sub(r"\s+", " ", nfc(line.strip()).lower())
        if not w or len(w) > a.max_chars or w.count(" ") + 1 > a.max_syllables:
            continue
        if pat.match(w) and w not in seen:
            seen.add(w)
            kept.append(w)
    kept.sort()
    Path(a.out).write_text("\n".join(kept) + "\n", encoding="utf-8")

    n_diac = sum(any(c in VI_LOWER for c in w) for w in kept)
    print(f"Giữ {len(kept)} mục (từ {len(raw.splitlines())} dòng); {n_diac} mục có dấu/ă/â/ê/ô/ơ/ư/đ")
    print(f"Đã ghi {a.out}")
    if len(kept) < 10000:
        print("CẢNH BÁO: dưới 10.000 mục, nên bổ sung nguồn khác.")


if __name__ == "__main__":
    main()
