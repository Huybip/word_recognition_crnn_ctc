"""Test nghiệm thu Giai đoạn 1. Chạy: pytest -q tests/test_data.py
Dữ liệu lấy từ biến môi trường DATA_ROOT (mặc định data/synth). Thiếu dữ liệu -> bỏ qua test tương ứng."""
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import OCRDataset, collate_fn  # noqa: E402
from src.text_utils import VI_LETTERS, VI_LOWER, Charset, strip_diacritics  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
DATA = Path(os.environ.get("DATA_ROOT", ROOT / CFG["out_root"]))
LANGS = [l for l in ("en", "vi") if (DATA / l / "manifest.json").exists()]
MAIN = ["train", "val", "test", "hard"]  # train_clean cố ý chung nhãn với train

needs_data = pytest.mark.skipif(not LANGS, reason="chưa sinh dữ liệu (python data/generate_dataset.py)")
needs_charset = pytest.mark.skipif(not (ROOT / "data" / "charset.txt").exists(), reason="chưa chạy build_charset.py")


def read(lang, split):
    d = DATA / lang / split
    rows = [l.split("\t") for l in (d / "labels.tsv").read_text(encoding="utf-8").split("\n") if l]
    return d, rows


def splits_of(lang):
    return [s for s in MAIN if (DATA / lang / s / "labels.tsv").exists()]


# ---------------------------------------------------------------- tiện ích văn bản (luôn chạy)
def test_vi_letters_are_nfc_and_complete():
    assert unicodedata.normalize("NFC", VI_LETTERS) == VI_LETTERS
    assert len(VI_LOWER) == 67 and len(set(VI_LETTERS)) == 134


def test_strip_diacritics():
    assert strip_diacritics("Việt Nam Đẹp Ưu Ớt đ") == "Viet Nam Dep Uu Ot d"
    assert strip_diacritics("abc123") == "abc123"


def test_charset_roundtrip_and_blank(tmp_path):
    cs = Charset([" ", "a", "ệ", "Ư"])
    assert cs.num_classes == 5 and cs.encode("aệ Ư") == [2, 3, 1, 4]
    assert cs.decode(cs.encode("aệ Ư")) == "aệ Ư" and cs.decode([0, 2, 0]) == "a"
    cs.save(tmp_path / "c.txt")
    cs2 = Charset.load(tmp_path / "c.txt")
    assert cs2.chars == cs.chars and cs2.stoi[" "] == 1
    with pytest.raises(KeyError):
        cs.encode("z")


def test_encode_normalizes_nfd_to_nfc():
    cs = Charset(["ệ"])
    assert cs.encode(unicodedata.normalize("NFD", "ệ")) == [1]


# ---------------------------------------------------------------- dữ liệu
@needs_data
@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_nfc_and_clean(lang):
    for s in splits_of(lang) + ["train_clean"]:
        if not (DATA / lang / s / "labels.tsv").exists():
            continue
        _, rows = read(lang, s)
        for r in rows:
            assert len(r) == 3
            assert unicodedata.normalize("NFC", r[1]) == r[1], f"{lang}/{s}: nhãn không NFC: {r[1]!r}"
            assert r[1] == r[1].strip() and "  " not in r[1] and r[1] != ""


@needs_data
@pytest.mark.parametrize("lang", LANGS)
def test_splits_have_disjoint_labels(lang):
    sets = {s: {read(lang, s)[1][i][1].casefold() for i in range(len(read(lang, s)[1]))} for s in splits_of(lang)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"{lang}: {a} và {b} trùng nhãn: {list(sets[a] & sets[b])[:5]}"


@needs_data
@pytest.mark.parametrize("lang", LANGS)
def test_no_duplicate_images_across_splits(lang):
    owner = {}
    for s in splits_of(lang):
        d, rows = read(lang, s)
        for r in rows:
            h = hashlib.md5((d / r[0]).read_bytes()).hexdigest()
            assert owner.setdefault(h, s) == s, f"{lang}: ảnh {r[0]} của {s} trùng ảnh của {owner[h]}"


@needs_data
@needs_charset
@pytest.mark.parametrize("lang", LANGS)
def test_all_label_chars_in_charset(lang):
    cs = Charset.load(ROOT / "data" / "charset.txt")
    known = set(cs.chars)
    for s in splits_of(lang) + ["train_clean"]:
        if not (DATA / lang / s / "labels.tsv").exists():
            continue
        for r in read(lang, s)[1]:
            missing = set(r[1]) - known
            assert not missing, f"{lang}/{s}: ký tự ngoài charset {missing} trong {r[1]!r}"
            cs.encode(r[1])  # không ném lỗi


@needs_charset
def test_charset_structure():
    cs = Charset.load(ROOT / "data" / "charset.txt")
    en = Charset.load(ROOT / "data" / "charset_en.txt")
    assert cs.chars[: len(en)] == en.chars, "ký tự EN phải giữ nguyên index để fine-tune EN->VI"
    assert 0 not in cs.stoi.values() and cs.num_classes == len(cs.chars) + 1
    assert set(VI_LETTERS) <= set(cs.chars)
    assert " " in cs.chars


@needs_data
@pytest.mark.parametrize("lang", LANGS)
def test_hard_split_uses_unseen_fonts(lang):
    m = json.loads((DATA / lang / "manifest.json").read_text(encoding="utf-8"))
    if "hard" not in m["counts"] or set(m["holdout_fonts"]) == set(m["train_fonts"]):
        pytest.skip("không có font giữ riêng")
    hard_fonts = {r[2] for r in read(lang, "hard")[1]}
    other = {r[2] for s in ("train", "val", "test") for r in read(lang, s)[1]}
    assert hard_fonts and not (hard_fonts & other), f"font hard lọt sang tập khác: {hard_fonts & other}"


@needs_data
def test_vi_fonts_cover_all_label_chars():
    """Chống lỗi 'ô vuông': font dùng cho nhãn VI phải có glyph cho mọi ký tự của nhãn đó."""
    if "vi" not in LANGS:
        pytest.skip("không có dữ liệu vi")
    from fontTools.ttLib import TTFont
    paths = {Path(p).name: p for p in (ROOT / "data" / "fonts" / "fonts_vi.txt").read_text(encoding="utf-8").split("\n") if p}
    need = {}
    for s in splits_of("vi"):
        for r in read("vi", s)[1]:
            need.setdefault(r[2], set()).update(r[1])
    for font, chars in need.items():
        cmap = TTFont(paths[font], fontNumber=0, lazy=True).getBestCmap()
        miss = {c for c in chars if c != " " and ord(c) not in cmap}
        assert not miss, f"font {font} thiếu glyph {miss}"


@needs_data
def test_vi_has_diacritics_and_wordlist_size():
    if "vi" not in LANGS:
        pytest.skip("không có dữ liệu vi")
    m = json.loads((DATA / "vi" / "manifest.json").read_text(encoding="utf-8"))
    if not m["smoke"]:
        assert sum(m["vocab_sizes"].values()) >= 10000
    labels = " ".join(r[1] for r in read("vi", "train")[1])
    assert len(set(labels) & set(VI_LETTERS)) >= 40, "dữ liệu VI có quá ít chữ cái có dấu"


@needs_data
@needs_charset
@pytest.mark.parametrize("lang", LANGS)
def test_dataset_and_collate(lang):
    cs = Charset.load(ROOT / "data" / "charset.txt")
    H = CFG["img_height"]
    ds = OCRDataset(DATA / lang / "train", cs, img_height=H)
    items = [ds[i] for i in range(8)]
    for it in items:
        assert it["image"].shape[0:2] == (1, H) and str(it["image"].dtype) == "torch.float32"
        assert -1.001 <= float(it["image"].min()) and float(it["image"].max()) <= 1.001
        assert cs.decode(it["target"].tolist()) == it["text"]
    b = collate_fn(items)
    n, wmax = b["images"].shape[0], b["images"].shape[-1]
    assert n == 8 and b["images"].shape[1:3] == (1, H)
    assert b["targets"].numel() == int(b["target_lengths"].sum())
    assert int(b["widths"].max()) == wmax and (b["widths"] <= wmax).all()
    assert b["target_lengths"].tolist() == [len(t) for t in b["texts"]]
    assert (b["targets"] > 0).all(), "nhãn không được chứa index 0 (blank)"
