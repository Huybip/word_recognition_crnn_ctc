#!/usr/bin/env python
"""Sinh dữ liệu ảnh chữ bằng TextRecognitionDataGenerator (TRDG).

Quy trình (đảm bảo KHÔNG trùng nhãn giữa các tập):
  1. Lọc từ vựng -> chia TỪ VỰNG thành train/val/test/hard (không phải chia ảnh).
  2. Với mỗi tập, ghép các mục từ vựng của riêng tập đó thành dòng chữ; loại bỏ mọi
     chuỗi trùng (không phân biệt hoa/thường) với từ vựng hoặc nhãn của tập khác.
  3. Chọn font ngẫu nhiên; tập `hard` chỉ dùng font của các HỌ FONT bị giữ riêng (font lạ).
  4. Render bằng TRDG + augmentation theo mức none/normal/hard, lưu PNG xám.
Đầu ra: <out_root>/<lang>/<split>/{images/*.png, labels.tsv}, <out_root>/<lang>/manifest.json
labels.tsv: path <TAB> text <TAB> font
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compat  # noqa: E402,F401  (vá Pillow trước khi import trdg)
from src.text_utils import VI_LOWER, nfc  # noqa: E402

import trdg  # noqa: E402
from trdg.data_generator import FakeTextDataGenerator  # noqa: E402

TRDG_DIR = Path(trdg.__file__).parent
BG_IMAGE_DIR = str(TRDG_DIR / "images")
SPLITS = ["train", "val", "test", "hard", "train_clean"]

# ------------------------------------------------------------------ augmentation
# bg: 0 nhiễu Gauss, 1 nền trắng, 2 quasicrystal, 3 ảnh nền | dist: 0 không, 1 sin, 2 cos, 3 ngẫu nhiên
_NOPOST = dict(lowres_p=0, lowres_scale=(1, 1), contrast_p=0, contrast=(1, 1), bright_p=0, bright=(1, 1),
               noise_p=0, noise_sigma=(0, 0), sp_p=0, sp_amount=(0, 0), jpeg_p=0, jpeg_q=(90, 90))
AUG_LEVELS = {
    "none": dict(skew=0, blur=0.0, bg={1: 1.0}, dist={0: 1.0}, color="#000000,#222222", post=_NOPOST),
    "normal": dict(
        skew=2, blur=1.2, bg={0: 0.3, 1: 0.3, 2: 0.2, 3: 0.2}, dist={0: 0.6, 1: 0.2, 2: 0.2},
        color="#000000,#444444",
        post=dict(lowres_p=0.10, lowres_scale=(0.6, 0.9), contrast_p=0.4, contrast=(0.7, 1.2),
                  bright_p=0.4, bright=(0.8, 1.2), noise_p=0.3, noise_sigma=(2, 8),
                  sp_p=0.05, sp_amount=(0.002, 0.01), jpeg_p=0.3, jpeg_q=(40, 90))),
    "hard": dict(
        skew=5, blur=2.5, bg={0: 0.3, 2: 0.35, 3: 0.35}, dist={1: 0.3, 2: 0.3, 3: 0.4},
        color="#000000,#8A8A8A",
        post=dict(lowres_p=0.6, lowres_scale=(0.35, 0.6), contrast_p=0.8, contrast=(0.45, 0.9),
                  bright_p=0.5, bright=(0.7, 1.3), noise_p=0.85, noise_sigma=(10, 28),
                  sp_p=0.4, sp_amount=(0.01, 0.04), jpeg_p=0.7, jpeg_q=(10, 45))),
}


def wchoice(r: random.Random, d: dict):
    return r.choices(list(d.keys()), weights=list(d.values()))[0]


def post_aug(img: Image.Image, p: dict, r: random.Random) -> Image.Image:
    """Augmentation sau TRDG (numpy/PIL): low-res, tương phản, sáng, nhiễu, muối tiêu, JPEG."""
    if r.random() < p["lowres_p"]:
        s, (w, h) = r.uniform(*p["lowres_scale"]), img.size
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if r.random() < p["contrast_p"]:
        img = ImageEnhance.Contrast(img).enhance(r.uniform(*p["contrast"]))
    if r.random() < p["bright_p"]:
        img = ImageEnhance.Brightness(img).enhance(r.uniform(*p["bright"]))
    arr = np.asarray(img, dtype=np.float32)
    if r.random() < p["noise_p"]:
        arr = arr + np.random.normal(0, r.uniform(*p["noise_sigma"]), arr.shape)
    if r.random() < p["sp_p"]:
        amt, m = r.uniform(*p["sp_amount"]), np.random.rand(*arr.shape)
        arr[m < amt / 2], arr[m > 1 - amt / 2] = 0, 255
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if r.random() < p["jpeg_p"]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(r.uniform(*p["jpeg_q"])))
        img = Image.open(buf).convert("L")
    return img


def render_job(job: dict):
    """Worker: render 1 ảnh -> lưu PNG -> trả (path, text, font, md5)."""
    lvl = AUG_LEVELS[job["level"]]
    img = None
    for attempt in range(7):  # TRDG có thể trả None (chữ/nền quá giống màu) -> thử seed khác
        seed = job["seed"] + attempt * 7919
        random.seed(seed)  # TRDG dùng random toàn cục
        np.random.seed(seed % (2**32 - 1))
        r = random.Random(seed ^ 0x9E3779B1)
        last = attempt == 6
        bg = 1 if last else wchoice(r, lvl["bg"])
        dist = 0 if last else wchoice(r, lvl["dist"])
        with contextlib.redirect_stdout(io.StringIO()):  # TRDG in log rác
            img = FakeTextDataGenerator.generate(
                job["idx"], job["text"], job["font"], None, job["size"], None,
                lvl["skew"], True, lvl["blur"], True, bg, dist, r.randint(0, 2), False, 0, -1, 1,
                lvl["color"], 0, 1.0, 0, tuple(job["margins"]), False, False, False,
                BG_IMAGE_DIR, 0, "#282828", "RGB", 0)
        if img is not None:
            break
    if img is None:
        raise RuntimeError(f"Không render được: {job['text']!r} với {job['font']}")
    img = post_aug(img.convert("L"), lvl["post"], r)
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=3)
    data = buf.getvalue()
    Path(job["out_path"]).write_bytes(data)
    return job["rel_path"], job["text"], Path(job["font"]).name, hashlib.md5(data).hexdigest()


# ------------------------------------------------------------------ từ vựng & văn bản
def load_vocab(lang: str, cfg: dict) -> list:
    max_chars = cfg["text"]["max_chars"]
    if lang == "en":
        words = (TRDG_DIR / "dicts" / "en.txt").read_text(encoding="utf-8").split("\n")
        pat = re.compile(r"^[a-z]{2,14}$")
        entries = {w.lower() for w in (x.strip() for x in words) if pat.match(w.lower())}
    else:
        path = ROOT / cfg["vocab"]["vi_file"]
        if not path.exists():
            sys.exit(f"Thiếu {path}. Chạy trước: python data/make_vi_wordlist.py")
        entries = {nfc(x.strip()).lower() for x in path.read_text(encoding="utf-8").split("\n") if x.strip()}
    return sorted(e for e in entries if 0 < len(e) <= max_chars)


def split_vocab(entries: list, ratios: dict, seed: int) -> dict:
    rng = random.Random(seed)
    e = list(entries)
    rng.shuffle(e)
    n, out, start = len(e), {}, 0
    names = ["train", "val", "test", "hard"]
    for i, s in enumerate(names):
        end = n if i == len(names) - 1 else start + int(n * ratios[s])
        out[s] = e[start:end]
        start = end
    return out


def apply_case(r: random.Random, s: str, weights: dict) -> str:
    m = wchoice(r, weights)
    if m == "capitalize":
        s = s[:1].upper() + s[1:]
    elif m == "upper":
        s = s.upper()
    elif m == "title":
        s = s.title()
    return nfc(s)


CONF = "O0Il1S5B8Z2G6"


def make_code(r: random.Random) -> str:
    k = r.choice(["digits", "mixed", "confus", "price", "date", "dashed"])
    if k == "digits":
        return "".join(r.choices("0123456789", k=r.randint(3, 10)))
    if k == "mixed":
        return "".join(r.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=r.randint(4, 10)))
    if k == "confus":
        return "".join(r.choices(CONF + "ADEKMRTWab", k=r.randint(4, 9)))
    if k == "price":
        return f"{r.randint(1, 99)}.{r.randint(0, 999):03d}.{r.randint(0, 999):03d}"
    if k == "date":
        return f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(1990, 2030)}"
    return "".join(r.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=r.randint(2, 3))) + "-" + "".join(r.choices("0123456789", k=r.randint(3, 5)))


def build_texts(lang, cfg, pools, counts, seed):
    """Trả {split: [text,...]}; bảo đảm nhãn các tập train/val/test/hard rời nhau."""
    tc = cfg["text"]
    w = tc["n_words_weights"][lang]
    key = lambda t: nfc(t).casefold()  # noqa: E731
    pool_keys = {s: {key(x) for x in pools[s]} for s in pools}
    made = {}
    for s in ["test", "hard", "val", "train"]:  # tập "kín" sinh trước
        n = counts[s]
        if n <= 0:
            made[s] = []
            continue
        rng = random.Random(seed * 31 + len(s) * 1009 + n)
        forbid = set().union(*(pool_keys[o] for o in pools if o != s))
        for o in made:
            forbid |= {key(t) for t in made[o]}
        pool, texts, seen, tries = pools[s], [], set(), 0
        unique = s != "train"
        while len(texts) < n:
            tries += 1
            if tries > 50 * n + 1000:
                raise RuntimeError(f"Không đủ nhãn khác nhau cho tập {s} ({len(texts)}/{n}); "
                                   f"giảm số lượng hoặc bổ sung từ vựng.")
            if rng.random() < tc["code_ratio"]:
                t = make_code(rng)
            else:
                for _ in range(20):
                    k = rng.choices(range(1, len(w) + 1), weights=w)[0]
                    t = " ".join(rng.choice(pool) for _ in range(k))
                    if len(t) <= tc["max_chars"]:
                        break
                else:
                    t = rng.choice(pool)
                t = apply_case(rng, t, tc["case_weights"])
            kk = key(t)
            if kk in forbid or (unique and kk in seen):
                continue
            seen.add(kk)
            texts.append(t)
        made[s] = texts
    return made


# ------------------------------------------------------------------ font
def font_family(name: str) -> str:
    base = re.split(r"[-_.]", Path(name).stem + "-")[0]
    return re.sub(r"(BoldOblique|BoldItalic|Bold|Italic|Oblique)+$", "", base) or base


def split_fonts(fonts: list, ratio: float, seed: int):
    """Giữ riêng nguyên HỌ FONT (không chỉ vài kiểu của cùng họ) để tập hard có font lạ thật."""
    fams = sorted({font_family(f) for f in fonts})
    random.Random(seed).shuffle(fams)
    n_hold = max(1, round(len(fams) * ratio)) if len(fams) > 1 else 0
    hold = set(fams[:n_hold])
    train = [f for f in fonts if font_family(f) not in hold]
    held = [f for f in fonts if font_family(f) in hold]
    return train, held, sorted(hold)


# ------------------------------------------------------------------ chính
def generate_lang(lang, cfg, args):
    seed = args.seed if args.seed is not None else cfg["seed"]
    counts = dict(cfg["smoke_counts"] if args.smoke else cfg["counts"][lang])
    out_root = Path(args.out or cfg["out_root"])
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    lang_dir = out_root / lang

    flist = ROOT / "data" / "fonts" / f"fonts_{lang}.txt"
    if not flist.exists():
        sys.exit("Thiếu danh sách font. Chạy trước: python data/check_fonts.py")
    fonts = [x for x in flist.read_text(encoding="utf-8").split("\n") if x]
    train_fonts, hold_fonts, hold_fams = split_fonts(fonts, cfg["fonts"]["holdout_ratio"], seed)
    if not hold_fonts:
        print("CẢNH BÁO: không có font giữ riêng; tập hard sẽ dùng chung font với train.")
        hold_fonts = train_fonts
    print(f"[{lang}] {len(fonts)} font | train/val/test: {len(train_fonts)} | hard (họ font lạ {hold_fams}): {len(hold_fonts)}")

    vocab = load_vocab(lang, cfg)
    pools = split_vocab(vocab, cfg["vocab"]["split_ratio"], seed)
    print(f"[{lang}] từ vựng {len(vocab)} -> " + ", ".join(f"{s}={len(v)}" for s, v in pools.items()))
    if lang == "vi" and len(vocab) < 10000:
        print("CẢNH BÁO: từ điển VI dưới 10.000 mục.")

    texts = build_texts(lang, cfg, pools, counts, seed)
    texts["train_clean"] = texts["train"][: counts.get("train_clean", 0)]

    if lang_dir.exists():
        if not args.overwrite:
            sys.exit(f"{lang_dir} đã tồn tại. Dùng --overwrite để sinh lại.")
        shutil.rmtree(lang_dir)

    rng = random.Random(seed + 7)
    size, margins = cfg["render"]["size"], cfg["render"]["margins"]
    manifest = dict(lang=lang, seed=seed, smoke=args.smoke, counts={}, render=cfg["render"],
                    train_fonts=[Path(f).name for f in train_fonts],
                    holdout_fonts=[Path(f).name for f in hold_fonts], holdout_families=hold_fams,
                    vocab_sizes={s: len(v) for s, v in pools.items()}, aug_level=cfg["aug_level"])
    md5_owner, dup_cross = {}, 0
    for si, split in enumerate(SPLITS):
        tl = texts.get(split, [])
        if not tl:
            continue
        sdir = lang_dir / split
        (sdir / "images").mkdir(parents=True, exist_ok=True)
        font_pool = hold_fonts if split == "hard" else train_fonts
        jobs = []
        for i, t in enumerate(tl):
            rel = f"images/{i:06d}.png"
            jobs.append(dict(idx=i, text=t, font=rng.choice(font_pool), seed=(seed * 1_000_003 + (0 if lang == "en" else 17) * 7_919 + si * 100_003 + i) % (2**31),
                             level=cfg["aug_level"][split], size=size, margins=margins,
                             rel_path=rel, out_path=str(sdir / rel)))
        rows = []
        with Pool(args.workers) as pool:
            for rel, text, font, md5 in tqdm(pool.imap(render_job, jobs, chunksize=32), total=len(jobs), desc=f"{lang}/{split}"):
                rows.append((rel, text, font))
                if split != "train_clean":  # train_clean cố ý chung nhãn với train nên bỏ qua
                    owner = md5_owner.setdefault(md5, split)
                    dup_cross += owner != split
        with open(sdir / "labels.tsv", "w", encoding="utf-8") as f:
            for rel, text, font in rows:
                f.write(f"{rel}\t{text}\t{font}\n")
        manifest["counts"][split] = len(rows)
    manifest["duplicate_images_across_splits"] = dup_cross
    (lang_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{lang}] xong: {manifest['counts']} | ảnh trùng giữa các tập: {dup_cross}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--lang", choices=["en", "vi", "both"], default="both")
    ap.add_argument("--smoke", action="store_true", help="sinh bộ nhỏ (smoke_counts) để kiểm tra nhanh")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for lang in (["en", "vi"] if args.lang == "both" else [args.lang]):
        generate_lang(lang, cfg, args)


if __name__ == "__main__":
    main()
