"""Test Giai đoạn 5. Chạy: pytest -q tests/test_analyze.py"""
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analyze_errors import (CATS, aggregate, bootstrap_ci, classify_sub, decompose, error_events,  # noqa: E402
                                font_family, load_records, pick_examples, recommendations, write_report)
from src.metrics import corpus_metrics  # noqa: E402


# ---------------------------------------------------------------- phân rã NFD
def test_decompose():
    assert decompose("ố") == ("o", "mũ", "sắc")
    assert decompose("ơ") == ("o", "móc", None) and decompose("ữ") == ("u", "móc", "ngã")
    assert decompose("ặ") == ("a", "trăng", "nặng") and decompose("ệ") == ("e", "mũ", "nặng")
    assert decompose("đ") == ("d", "gạch", None) and decompose("Đ") == ("D", "gạch", None)
    assert decompose("a") == ("a", None, None) and decompose("Ấ") == ("A", "mũ", "sắc")
    assert decompose("5") is None and decompose(" ") is None and decompose("ü") is None


@pytest.mark.parametrize("r,h,cat", [
    ("á", "à", "tone_wrong"), ("ả", "ã", "tone_wrong"), ("á", "a", "tone_missing"), ("a", "á", "tone_extra"),
    ("ô", "o", "modifier_missing"), ("o", "ô", "modifier_extra"), ("â", "ă", "modifier_wrong"), ("ơ", "ô", "modifier_wrong"),
    ("đ", "d", "modifier_missing"), ("d", "đ", "modifier_extra"),
    ("ấ", "a", "diacritic_compound"), ("ấ", "ă", "diacritic_compound"), ("ồ", "ố", "tone_wrong"),
    ("ế", "ề", "tone_wrong"), ("ế", "ê", "tone_missing"),
    ("O", "0", "lookalike"), ("0", "O", "lookalike"), ("l", "1", "lookalike"), ("I", "l", "lookalike"), ("S", "5", "lookalike"),
    ("A", "a", "case"), ("Ấ", "ấ", "case"), ("v", "V", "case"),
    ("a", " ", "space_sub"), ("ê", "ô", "other_sub"), ("a", "b", "other_sub"),
])
def test_classify_sub(r, h, cat):
    assert classify_sub(r, h) == cat


def test_events_digraph_segment_and_deletion():
    ev, seg, _ = error_events("mama", "rnama")                    # m -> rn: một cụm hai thao tác
    assert [e["cat"] for e in ev] == ["lookalike_digraph"] * 2 and seg == [("m", "rn")]
    ev, seg, _ = error_events("Việt Nam", "Viet Nam")
    assert [(e["op"], e["cat"]) for e in ev] == [("sub", "diacritic_compound")]
    assert [e["cat"] for e in error_events("abc", "ab")[0]] == ["del_char"]
    assert [e["cat"] for e in error_events("abệ", "ab")[0]] == ["del_vi_diacritic_letter"]
    assert [e["cat"] for e in error_events("a b", "ab")[0]] == ["space_del"]
    assert [e["cat"] for e in error_events("ab", "a b")[0]] == ["space_ins"]
    assert [e["cat"] for e in error_events("ab", "abx")[0]] == ["ins_char"]
    assert error_events("abc", "abc")[0] == []


def test_every_edit_gets_exactly_one_category_property():
    """Tổng số lỗi theo loại phải bằng S+D+I (= khoảng cách Levenshtein) với mọi cặp chuỗi."""
    rng = random.Random(0)
    alpha = "abcdoOl01Sệấơư mnrđ"
    for _ in range(400):
        r = "".join(rng.choices(alpha, k=rng.randint(0, 14)))
        h = "".join(rng.choices(alpha, k=rng.randint(0, 14)))
        ev, _s, eq = error_events(r, h)
        m = corpus_metrics([r], [h])
        assert len(ev) == m["S"] + m["D"] + m["I"], (r, h)
        assert all(e["cat"] in CATS for e in ev)
        assert len(eq) + sum(e["op"] in ("sub", "del") for e in ev) == len(r)   # mọi ký tự nhãn đều được tính


# ---------------------------------------------------------------- tổng hợp
def make_records():
    pairs = [("Việt Nam", "Viet Nam"), ("Hà Nội", "Hà Nội"), ("0123", "O123"), ("mama", "rnama"), ("hello world", "helloworld"),
             ("ĐỔI MỚI", "ĐỔI MOI"), ("abc", "abc"), ("xin chào", "xin chao"), ("cổng", "cống"), ("Ngọc", "ngọc")]
    return [dict(path=f"images/{i}.png", ref=r, hyp=h, font=f"Lato-Regular.ttf" if i % 2 else "Roboto-Bold.ttf")
            for i, (r, h) in enumerate(pairs)]


def test_aggregate_numbers():
    res = aggregate(make_records(), n_boot=200)
    assert res["n"] == 10 and res["total_err"] == res["S"] + res["D"] + res["I"]
    assert sum(res["cat"].values()) == res["total_err"] and sum(res["group"].values()) == res["total_err"]
    assert res["cat"]["lookalike"] == 1 and res["cat"]["lookalike_digraph"] == 2 and res["cat"]["case"] == 1
    assert res["cat"]["space_del"] == 1 and res["cat"]["tone_wrong"] == 1                    # cổng -> cống (ổ -> ố)
    assert res["cat"]["tone_missing"] == 1                                                    # chào -> chao (à -> a)
    assert res["cat"]["diacritic_compound"] == 2                                              # Việt -> Viet (ệ->e), MỚI -> MOI (Ớ->O)
    assert res["total_err"] == 9
    assert res["word_acc"] == pytest.approx(0.2)   # chỉ "Hà Nội" và "abc" khớp hoàn toàn
    assert res["ci_cer"][0] <= res["cer"] <= res["ci_cer"][1] and res["ci_word"][0] <= res["word_acc"] <= res["ci_word"][1]
    assert res["pairs"][("0", "O")] == 1 and res["segs"][("m", "rn")] == 1


def test_ladder_strip_diacritics_reduces_cer():
    recs = [dict(path="a", ref="Việt Nam", hyp="Viet Nam"), dict(path="b", ref="xin chào", hyp="xin chao")]
    lad = aggregate(recs, n_boot=50)["ladder"]
    assert lad["đầy đủ"]["cer"] == pytest.approx(2 / 16) and lad["bỏ dấu"]["cer"] == 0
    assert lad["đầy đủ"]["word_acc"] == 0 and lad["bỏ dấu"]["word_acc"] == 1
    rec2 = [dict(path="a", ref="Abc", hyp="abc")]
    assert aggregate(rec2, n_boot=50)["ladder"]["bỏ hoa/thường"]["cer"] == 0


def test_no_diacritics_in_english_data_is_graceful(tmp_path):
    recs = [dict(path="a", ref="hello", hyp="hallo"), dict(path="b", ref="world", hyp="world")]
    res = aggregate(recs, n_boot=50)
    assert res["n_diac_chars"] == 0
    write_report(res, tmp_path, None, "en", 2)
    assert "không có chữ nào mang dấu" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_bootstrap_ci_deterministic_and_covers():
    e, n, x = [1, 0, 2, 0, 1, 0, 0, 3], [10] * 8, [0, 1, 0, 1, 0, 1, 1, 0]
    a = bootstrap_ci(e, n, x, 500, seed=1)
    assert a == bootstrap_ci(e, n, x, 500, seed=1)
    assert a[0][0] <= sum(e) / sum(n) <= a[0][1] and a[1][0] <= np.mean(x) <= a[1][1]
    assert bootstrap_ci([0] * 5, [5] * 5, [1] * 5, 100)[0] == (0.0, 0.0)


def test_font_family():
    assert font_family("RobotoCondensed-BoldItalic.ttf") == "RobotoCondensed" and font_family("FreeSerifBoldItalic.ttf") == "FreeSerif"
    assert font_family("Lato-Regular.ttf") == "Lato"


# ---------------------------------------------------------------- ví dụ & khuyến nghị
def test_pick_examples_stratified_and_deterministic():
    res = aggregate([dict(r) for r in make_records() * 3], n_boot=50)   # bản sao: tránh dùng chung một dict
    c1, w1 = pick_examples(res, 10, seed=0)
    c2, w2 = pick_examples(res, 10, seed=0)
    assert [r["path"] for r in w1] == [r["path"] for r in w2] and all(r["ref"] == r["hyp"] for r in c1)
    assert all(r["ref"] != r["hyp"] for r in w1) and len(w1) == 10 == len({id(r) for r in w1})
    cats_in_wrong = {c for r in w1 for c in r["cats"]}
    assert len(cats_in_wrong) >= 4                                  # phủ nhiều loại lỗi, không dồn vào một loại


def test_recommendations_cite_numbers_and_flag_unconverged():
    res = aggregate(make_records(), n_boot=50)
    txt = "\n".join(recommendations(res))
    assert "Lỗi dấu" in txt and "%" in txt                           # có đề xuất về dấu, có số liệu
    bad = aggregate([dict(path="a", ref="abcdefgh", hyp="")] * 40, n_boot=50)
    rec = recommendations(bad)
    assert rec[0].startswith("**CER của tập này rất cao") and "val CER" in rec[0] and any("Mất ký tự" in r for r in rec)
    bad["ref_val_cer"] = 0.07                      # mô hình tốt trên val -> khoảng cách khái quát hóa, KHÔNG phải chưa hội tụ
    assert recommendations(bad)[0].startswith("**Ưu tiên 1: khoảng cách khái quát hóa") and "không phải thiếu huấn luyện" in recommendations(bad)[0]
    bad["ref_val_cer"] = 0.95                      # val CER cũng cao -> có thể chưa hội tụ
    assert recommendations(bad)[0].startswith("**Ưu tiên 1: mô hình có thể chưa hội tụ")
    clean = aggregate([dict(path="a", ref="abc", hyp="abc")] * 50, n_boot=50)
    assert recommendations(clean) == ["Không có lỗi nào ở tập này nên không có gì để đề xuất."]
    one = aggregate([dict(path=f"{i}", ref="abcdefgh", hyp="abcdafgh" if i == 0 else "abcdefgh") for i in range(50)], n_boot=50)
    rec1 = recommendations(one)                      # 1 lỗi "thay thế khác": không quy tắc nào kích hoạt
    assert rec1[0].startswith("Không có tín hiệu") and "Chỉ có 1 lỗi" in rec1[-1]


# ---------------------------------------------------------------- end-to-end với ảnh giả
def test_report_end_to_end_with_images(tmp_path):
    from PIL import Image, ImageDraw
    d = tmp_path / "set"
    (d / "images").mkdir(parents=True)
    rng = random.Random(0)
    recs = [dict(r) for r in make_records() * 3]
    lines, tsv = [], ["path\tref\thyp\tedit_distance"]
    for i, r in enumerate(recs):
        im = Image.new("L", (rng.randint(60, 200), 40), 255)
        ImageDraw.Draw(im).text((3, 10), r["ref"], fill=rng.randint(0, 120))
        im.save(d / "images" / f"{i:03d}.png")
        lines.append(f"images/{i:03d}.png\t{r['ref']}\t{r['font']}")
        tsv.append(f"images/{i:03d}.png\t{r['ref']}\t{r['hyp']}\t0")
    (d / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "preds_vi_test_greedy.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    loaded = load_records(tmp_path / "preds_vi_test_greedy.tsv", d, 32)
    assert len(loaded) == 30 and loaded[0]["font"] == "Roboto-Bold.ttf" and loaded[0]["width"] >= 16 and loaded[0]["contrast"] > 0
    res = aggregate(loaded, 32, 4, 100, 0, min_font_n=5)
    out = tmp_path / "out"
    write_report(res, out, d, "test", 10, 0)
    md = (out / "report.md").read_text(encoding="utf-8")
    for sec in ["## 1. Tổng quan", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9."]:
        assert sec in md
    for f in ["error_types.png", "confusion_matrix.png", "tone_confusion.png", "cer_ladder.png", "by_length.png",
              "examples_correct.png", "examples_wrong.png"]:
        assert (out / "figures" / f).stat().st_size > 1000, f
    assert len(list((out / "examples").glob("wrong_*.png"))) >= 5 and len(list((out / "examples").glob("correct_*.png"))) >= 5
