#!/usr/bin/env python
"""Kiểm tra font: loại font thiếu glyph hoặc render ra ô trống.

Với mỗi font:
  1) cmap (fontTools) phải chứa mọi ký tự yêu cầu;
  2) PIL phải load được và mỗi ký tự phải render ra pixel (bbox khác rỗng).
Kết quả: data/fonts/fonts_en.txt, data/fonts/fonts_vi.txt, data/fonts/font_report.csv
và (tùy chọn --sheet) ảnh kiểm tra mắt thường outputs/figures/font_check_vi.png.
"""
import argparse
import csv
import re
import string
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compat  # noqa: E402,F401
from src.text_utils import VI_LETTERS  # noqa: E402

from fontTools.ttLib import TTFont  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

EN_NEED = string.ascii_letters + string.digits + ".,-/:()'"
VI_NEED = EN_NEED + VI_LETTERS
EXTS = {".ttf", ".otf", ".ttc"}


def collect_fonts(cfg_fonts):
    dirs = []
    if cfg_fonts.get("use_trdg_fonts", True):
        import trdg
        dirs.append(Path(trdg.__file__).parent / "fonts" / "latin")
    for d in list(cfg_fonts.get("extra_dirs", [])) + list(cfg_fonts.get("system_dirs", [])):
        p = Path(d)
        dirs.append(p if p.is_absolute() else ROOT / p)
    files = []
    for d in dirs:
        if d.exists():
            files += [p for p in d.rglob("*") if p.suffix.lower() in EXTS]
    exc = cfg_fonts.get("exclude_regex") or None
    out, seen = [], set()
    for p in sorted(files):
        if exc and re.search(exc, p.name):
            continue
        if p.name in seen:  # tránh trùng tên (test tra cứu font theo basename)
            continue
        seen.add(p.name)
        out.append(p)
    return out


def check_font(path: Path, need: str):
    """Trả về danh sách ký tự bị thiếu/render rỗng."""
    try:
        cmap = TTFont(str(path), fontNumber=0, lazy=True).getBestCmap() or {}
        font = ImageFont.truetype(str(path), 48)
    except Exception:
        return list(need)
    bad = []
    for ch in dict.fromkeys(need):
        if ord(ch) not in cmap:
            bad.append(ch)
            continue
        try:
            bb = font.getbbox(ch)
        except Exception:
            bad.append(ch)
            continue
        if not bb or bb[2] - bb[0] <= 0 or bb[3] - bb[1] <= 0:
            bad.append(ch)
    return bad


def make_sheet(fonts, out_png, text="Tiếng Việt: ổ ữ ặ ẫ ệ Ở Ự Ấ Ệ Ẫ Đ đ 0O1lI"):
    fonts = fonts[:48]
    h = 44
    img = Image.new("RGB", (900, h * len(fonts) + 10), "white")
    d = ImageDraw.Draw(img)
    label_font = ImageFont.load_default()
    for i, p in enumerate(fonts):
        y = 5 + i * h
        d.text((5, y + 14), p.name[:22], fill="gray", font=label_font)
        d.text((190, y), text, fill="black", font=ImageFont.truetype(str(p), 30))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--sheet", action="store_true", help="xuất ảnh kiểm tra font tiếng Việt")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))

    fonts = collect_fonts(cfg["fonts"])
    rows, ok_en, ok_vi = [], [], []
    for p in fonts:
        miss_vi = check_font(p, VI_NEED)
        miss_en = [c for c in miss_vi if c in EN_NEED] if len(miss_vi) < len(VI_NEED) else list(EN_NEED)
        if not miss_en:
            ok_en.append(p)
        if not miss_vi:
            ok_vi.append(p)
        rows.append([str(p), not miss_en, not miss_vi, len(miss_vi), "".join(miss_vi[:12])])

    out = ROOT / "data" / "fonts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fonts_en.txt").write_text("\n".join(map(str, ok_en)) + "\n", encoding="utf-8")
    (out / "fonts_vi.txt").write_text("\n".join(map(str, ok_vi)) + "\n", encoding="utf-8")
    with open(out / "font_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "ok_en", "ok_vi", "n_missing_vi", "missing_vi_sample"])
        w.writerows(rows)

    print(f"Tổng font quét: {len(fonts)} | dùng được cho EN: {len(ok_en)} | đủ dấu tiếng Việt: {len(ok_vi)}")
    n_min = cfg["fonts"].get("min_fonts_warn", 6)
    for name, lst in (("EN", ok_en), ("VI", ok_vi)):
        if len(lst) < n_min:
            print(f"CẢNH BÁO: chỉ {len(lst)} font {name}; thêm font vào data/fonts_extra hoặc system_dirs.")
    if a.sheet and ok_vi:
        png = ROOT / "outputs" / "figures" / "font_check_vi.png"
        make_sheet(ok_vi, png)
        print(f"Ảnh kiểm tra font: {png}")


if __name__ == "__main__":
    main()
