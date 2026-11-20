#!/usr/bin/env python
"""Phân tích lỗi nhận dạng từ file dự đoán của eval.py (preds_<tập>_<decoder>.tsv).

Ví dụ:
  python -m src.analyze_errors --tag vi_finetune --set vi_test --decoder greedy
  python -m src.analyze_errors --preds outputs/eval/x/preds_vi_test_greedy.tsv --data_dir data/synth/vi/test --out outputs/analysis/x

Mỗi lỗi (một thao tác thay/xóa/chèn của căn chỉnh Levenshtein) được gán đúng MỘT loại, nên
tổng số lỗi theo loại luôn bằng S+D+I. Hai cách căn chỉnh tối thiểu khác nhau có thể cho cặp nhầm lẫn
khác nhau; hàm align() dùng quy tắc hòa cố định (khớp/thay > xóa > chèn) để kết quả tái lập được.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ctc_check import ctc_min_length
from .metrics import align, corpus_metrics
from .text_utils import strip_diacritics

ROOT = Path(__file__).resolve().parents[1]

# ====================================================================== phân rã chữ có dấu tiếng Việt
TONES = {"\u0300": "huyền", "\u0301": "sắc", "\u0309": "hỏi", "\u0303": "ngã", "\u0323": "nặng"}
MODS = {"\u0306": "trăng", "\u0302": "mũ", "\u031B": "móc"}
TONE_ORDER = ["không", "sắc", "huyền", "hỏi", "ngã", "nặng"]


def decompose(ch: str):
    """(chữ_gốc, dấu_phụ, dấu_thanh) bằng NFD; None nếu không phải chữ Latin (có/không dấu tiếng Việt).
    dấu_phụ in {None,'trăng','mũ','móc','gạch'}; dấu_thanh in {None,'sắc','huyền','hỏi','ngã','nặng'}.
    'đ' không phân rã được trong NFD nên xử lý riêng: gốc 'd', dấu phụ 'gạch'."""
    if ch in ("đ", "Đ"):
        return ("d" if ch == "đ" else "D"), "gạch", None
    d = unicodedata.normalize("NFD", ch)
    base, marks = d[0], d[1:]
    if not (base.isascii() and base.isalpha()):
        return None
    mod = tone = None
    for m in marks:
        if m in TONES:
            tone = TONES[m]
        elif m in MODS:
            mod = MODS[m]
        else:
            return None
    return base, mod, tone


def has_diacritic(ch: str) -> bool:
    d = decompose(ch)
    return bool(d and (d[1] or d[2]))


# ====================================================================== phân loại lỗi
LOOKALIKE_GROUPS = ["0Oo", "1lI", "5Ss", "8B", "2Zz", "6G", "9gq", "uv", "il", "mn", "ec"]  # có thể chỉnh sửa
LOOKALIKE = {frozenset(p) for g in LOOKALIKE_GROUPS for p in ((a, b) for a in g for b in g if a != b)}
DIGRAPHS = {("m", "rn"), ("rn", "m"), ("d", "cl"), ("cl", "d"), ("w", "vv"), ("vv", "w"),
            ("m", "nn"), ("nn", "m"), ("h", "ln"), ("ln", "h"), ("u", "ii"), ("ii", "u")}

CATS = {  # mã loại -> (tên hiển thị, nhóm lớn)
    "tone_wrong": ("Sai dấu thanh (có dấu nhưng sai loại)", "diacritic"),
    "tone_missing": ("Thiếu dấu thanh", "diacritic"),
    "tone_extra": ("Thừa dấu thanh", "diacritic"),
    "modifier_wrong": ("Sai dấu phụ (mũ/móc/trăng nhầm nhau)", "diacritic"),
    "modifier_missing": ("Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc)", "diacritic"),
    "modifier_extra": ("Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ)", "diacritic"),
    "diacritic_compound": ("Sai cả dấu phụ lẫn dấu thanh", "diacritic"),
    "del_vi_diacritic_letter": ("Mất hẳn chữ có dấu", "deletion"),
    "lookalike": ("Ký tự giống nhau (O/0, l/1/I, S/5...)", "lookalike"),
    "lookalike_digraph": ("Cặp ký tự giống nhau (rn/m, cl/d...)", "lookalike"),
    "case": ("Sai hoa/thường (c/C, o/O...)", "case"),
    "space_sub": ("Khoảng trắng bị thay", "space"),
    "space_del": ("Mất khoảng trắng", "space"),
    "space_ins": ("Thừa khoảng trắng", "space"),
    "del_char": ("Mất ký tự", "deletion"),
    "ins_char": ("Thừa ký tự", "insertion"),
    "other_sub": ("Thay thế khác", "other"),
}
GROUP_NAMES = {"diacritic": "Lỗi dấu tiếng Việt", "lookalike": "Ký tự giống nhau", "case": "Hoa/thường",
               "space": "Khoảng trắng", "deletion": "Mất ký tự", "insertion": "Thừa ký tự", "other": "Thay thế khác"}
CAUSE = {  # mô tả cơ chế khả dĩ; là GIẢ THUYẾT dựa trên loại lỗi, chưa kiểm chứng bằng thí nghiệm
    "tone_wrong": "dấu thanh là nét nhỏ, dễ bị nhầm loại khi ảnh bị thu nhỏ hoặc nhiễu",
    "tone_missing": "dấu thanh mảnh bị mờ/mất sau khi resize hoặc bị nền nhiễu che",
    "tone_extra": "nhiễu/nét nền bị đọc thành dấu thanh",
    "modifier_wrong": "dấu mũ, móc, trăng có hình gần giống nhau ở độ phân giải thấp",
    "modifier_missing": "dấu mũ/móc/trăng nhỏ, gần như trùng hình với chữ gốc ở độ phân giải thấp",
    "modifier_extra": "nét nhỏ trên chữ bị đọc thành dấu phụ",
    "diacritic_compound": "chữ có nhiều dấu chồng nhau, cả hai dấu đều bị mất hoặc sai",
    "del_vi_diacritic_letter": "chữ có dấu bị bỏ hẳn, thường do T hẹp, ảnh mờ hoặc mô hình thiên về blank",
    "lookalike": "hai ký tự có hình dạng gần giống nhau, cần ngữ cảnh để phân biệt",
    "lookalike_digraph": "hai ký tự liền nhau trông giống một ký tự khác (rn ~ m)",
    "case": "chữ hoa và thường cùng hình dạng, chỉ khác kích thước",
    "space_sub": "khoảng trắng hẹp hoặc rộng tùy font nên ranh giới từ khó xác định",
    "space_del": "khoảng trắng hẹp bị bỏ qua",
    "space_ins": "khoảng trống trong chữ bị đọc thành dấu cách",
    "del_char": "ký tự bị bỏ sót: T hẹp so với nhãn, nét mờ hoặc mô hình chưa hội tụ",
    "ins_char": "nhiễu nền hoặc nét đứt bị đọc thành ký tự thừa",
    "other_sub": "nhầm hai ký tự không thuộc nhóm nào ở trên",
}


def classify_sub(r: str, h: str) -> str:
    if r == " " or h == " ":
        return "space_sub"
    if r != h and r.casefold() == h.casefold():
        return "case"
    dr, dh = decompose(r), decompose(h)
    if dr and dh and dr[0].casefold() == dh[0].casefold():  # cùng chữ gốc (không phân biệt hoa/thường)
        (_, mr, tr), (_, mh, th) = dr, dh
        if mr != mh and tr != th:
            return "diacritic_compound"
        if mr != mh:
            return "modifier_missing" if mh is None else "modifier_extra" if mr is None else "modifier_wrong"
        if tr != th:
            return "tone_missing" if th is None else "tone_extra" if tr is None else "tone_wrong"
    if frozenset((r, h)) in LOOKALIKE:
        return "lookalike"
    return "other_sub"


def classify_op(op: str, r: str, h: str) -> str:
    if op == "sub":
        return classify_sub(r, h)
    if op == "del":
        return "space_del" if r == " " else "del_vi_diacritic_letter" if has_diacritic(r) else "del_char"
    return "space_ins" if h == " " else "ins_char"


def error_events(ref: str, hyp: str):
    """Trả (events, segments, equal_chars).
    events  : mỗi thao tác lỗi một dict(op, r, h, cat). Tổng = S+D+I.
    segments: mỗi cụm lỗi liền nhau một (đoạn_nhãn, đoạn_dự_đoán), vd ('m','rn').
    Cụm khớp DIGRAPHS (m<->rn...) gán loại 'lookalike_digraph' cho mọi thao tác của cụm."""
    events, segments, equal_chars, seg = [], [], [], []

    def flush():
        if not seg:
            return
        rs = "".join(ref[i] for o, i, _j in seg if o in ("sub", "del"))
        hs = "".join(hyp[j] for o, _i, j in seg if o in ("sub", "ins"))
        is_dg = (rs.casefold(), hs.casefold()) in DIGRAPHS
        for o, i, j in seg:
            r = ref[i] if o != "ins" else ""
            h = hyp[j] if o != "del" else ""
            events.append(dict(op=o, r=r, h=h, cat="lookalike_digraph" if is_dg else classify_op(o, r, h)))
        segments.append((rs, hs))
        seg.clear()

    for o, i, j in align(ref, hyp):
        if o == "equal":
            flush()
            equal_chars.append(ref[i])
        else:
            seg.append((o, i, j))
    flush()
    return events, segments, equal_chars


# ====================================================================== đặc trưng mẫu
def font_family(name: str) -> str:
    base = re.split(r"[-_.]", Path(name).stem + "-")[0]
    return re.sub(r"(BoldOblique|BoldItalic|Bold|Italic|Oblique)+$", "", base) or base


def case_style(s: str) -> str:
    if any(c.isdigit() for c in s):
        return "có chữ số/mã"
    if s.isupper():
        return "CHỮ HOA"
    if s.islower():
        return "chữ thường"
    if s.istitle():
        return "Viết Hoa Từng Từ"
    return "Hoa đầu câu/hỗn hợp"


LEN_BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 999)]


def len_bin(n: int) -> str:
    for lo, hi in LEN_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    return "0"


def ratio_bin(x: float) -> str:
    return "<1.5" if x < 1.5 else "1.5-2" if x < 2 else "2-3" if x < 3 else ">=3"


def bootstrap_ci(edits, nchars, exact, n_boot=1000, seed=0):
    """Khoảng tin cậy 95% (bootstrap theo mẫu) cho CER gộp và Word Accuracy."""
    e, n, x = np.asarray(edits, float), np.asarray(nchars, float), np.asarray(exact, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    cers = e[idx].sum(1) / np.maximum(1, n[idx].sum(1))
    return (float(np.percentile(cers, 2.5)), float(np.percentile(cers, 97.5))), \
           (float(np.percentile(x[idx].mean(1), 2.5)), float(np.percentile(x[idx].mean(1), 97.5)))


def group_stats(recs, keyfn, order=None, min_n=1):
    g = defaultdict(list)
    for r in recs:
        k = keyfn(r)
        if k is not None:
            g[k].append(r)
    rows = []
    for k, lst in g.items():
        if len(lst) < min_n:
            continue
        N = sum(len(r["ref"]) for r in lst)
        E = sum(r["edits"] for r in lst)
        rows.append(dict(key=k, n=len(lst), cer=E / max(1, N), word_acc=sum(r["ref"] == r["hyp"] for r in lst) / len(lst)))
    if order:
        rows.sort(key=lambda d: order.index(d["key"]) if d["key"] in order else 99)
    else:
        rows.sort(key=lambda d: -d["cer"])
    return rows


# ====================================================================== phân tích tổng hợp
def aggregate(records, img_height=32, stride=4, n_boot=1000, seed=0, min_font_n=10):
    """records: list dict(path, ref, hyp, [font, width, contrast]). Trả về dict kết quả (không vẽ, không ghi file)."""
    refs, hyps = [r["ref"] for r in records], [r["hyp"] for r in records]
    cm = corpus_metrics(refs, hyps)
    cat, pairs, ref_chars, equal_chars = Counter(), Counter(), Counter(), Counter()
    segs, tone_mat, mod_detail = Counter(), Counter(), Counter()
    for rec in records:
        ev, sg, eq = error_events(rec["ref"], rec["hyp"])
        rec["events"], rec["cats"], rec["edits"] = ev, Counter(e["cat"] for e in ev), len(ev)
        ref_chars.update(rec["ref"])
        equal_chars.update(eq)
        for e in ev:
            cat[e["cat"]] += 1
            pairs[(e["r"] or "∅", e["h"] or "∅")] += 1
            if e["cat"].startswith("tone_"):
                tr, th = decompose(e["r"])[2] or "không", decompose(e["h"])[2] or "không"
                tone_mat[(tr, th)] += 1
            if e["cat"].startswith("modifier_"):
                mr, mh = decompose(e["r"])[1], decompose(e["h"])[1]
                mod_detail[(e["cat"], mr or "∅", mh or "∅")] += 1
        segs.update(s for s in sg if s != ("", ""))
        L = len(rec["ref"])
        rec["T_ratio"] = None
        if rec.get("width"):
            rec["T_ratio"] = (rec["width"] // stride) / max(1, ctc_min_length(rec["ref"])) if L else None
    total_err = sum(cat.values())
    assert total_err == cm["S"] + cm["D"] + cm["I"], "tổng lỗi theo loại phải bằng S+D+I"
    group = Counter()
    for c, n in cat.items():
        group[CATS[c][1]] += n
    ci_cer, ci_word = bootstrap_ci([r["edits"] for r in records], [len(r["ref"]) for r in records],
                                   [r["ref"] == r["hyp"] for r in records], n_boot, seed)
    # thang đo CER: bỏ dấu / bỏ phân biệt hoa-thường
    tf = {"đầy đủ": lambda s: s, "bỏ dấu": strip_diacritics, "bỏ hoa/thường": str.casefold,
          "bỏ dấu + hoa/thường": lambda s: strip_diacritics(s).casefold()}
    ladder = {}
    for k, f in tf.items():
        m = corpus_metrics([f(r) for r in refs], [f(h) for h in hyps])
        ladder[k] = dict(cer=m["cer"], word_acc=m["word_acc"])
    n_diac_chars = sum(1 for r in refs for c in r if has_diacritic(c))
    res = dict(n=cm["n"], n_chars=cm["n_chars"], S=cm["S"], D=cm["D"], I=cm["I"], cer=cm["cer"], word_acc=cm["word_acc"],
               ci_cer=ci_cer, ci_word=ci_word, total_err=total_err, cat=dict(cat), group=dict(group),
               pairs=pairs, ref_chars=ref_chars, equal_chars=equal_chars, segs=segs, tone_mat=tone_mat,
               mod_detail=mod_detail, ladder=ladder, n_diac_chars=n_diac_chars, records=records,
               img_height=img_height, stride=stride)
    res["by_length"] = group_stats(records, lambda r: len_bin(len(r["ref"])), [len_bin(b[0]) for b in LEN_BINS])
    res["by_case"] = group_stats(records, lambda r: case_style(r["ref"]) if r["ref"] else None)
    res["by_font"] = group_stats(records, lambda r: font_family(r["font"]) if r.get("font") else None, min_n=min_font_n)
    res["by_ratio"] = group_stats(records, lambda r: ratio_bin(r["T_ratio"]) if r.get("T_ratio") is not None else None,
                                  ["<1.5", "1.5-2", "2-3", ">=3"])
    con = [r["contrast"] for r in records if r.get("contrast") is not None]
    if con:
        q = np.quantile(con, [0.25, 0.5, 0.75])
        lab = lambda c: "Q1 (tương phản thấp nhất)" if c <= q[0] else "Q2" if c <= q[1] else "Q3" if c <= q[2] else "Q4 (cao nhất)"  # noqa: E731
        res["by_contrast"] = group_stats(records, lambda r: lab(r["contrast"]) if r.get("contrast") is not None else None,
                                         ["Q1 (tương phản thấp nhất)", "Q2", "Q3", "Q4 (cao nhất)"])
        res["contrast_q1"] = float(q[0])
    else:
        res["by_contrast"] = []
    res["mean_ratio_err"] = _mean([r["T_ratio"] for r in records if r.get("T_ratio") is not None and r["cats"]])
    res["mean_ratio_ok"] = _mean([r["T_ratio"] for r in records if r.get("T_ratio") is not None and not r["cats"]])
    res["mean_ratio_del"] = _mean([r["T_ratio"] for r in records if r.get("T_ratio") is not None
                                   and any(c.startswith("del") or c == "space_del" for c in r["cats"])])
    return res


def _mean(x):
    return float(np.mean(x)) if x else None


# ====================================================================== chọn ví dụ & nhận xét
def explain(rec, res) -> str:
    parts = []
    for c, _n in rec["cats"].most_common(3):
        parts.append(f"{CATS[c][0].lower()}: {CAUSE[c]}")
    ctx = []
    if rec.get("T_ratio") is not None and rec["T_ratio"] < 1.5:
        ctx.append(f"T/(L+R)={rec['T_ratio']:.2f} thấp (chuỗi thời gian hẹp so với nhãn)")
    if rec.get("contrast") is not None and res.get("contrast_q1") is not None and rec["contrast"] <= res["contrast_q1"]:
        ctx.append("ảnh thuộc nhóm tương phản thấp nhất 25%")
    if rec.get("font"):
        ctx.append(f"font {rec['font']}")
    return "; ".join(parts) + (" | bối cảnh đo được: " + ", ".join(ctx) if ctx else "")


def diff_str(rec, k=5) -> str:
    """Các ký tự khác nhau theo căn chỉnh, vd 'ụ→ự, n→∅' (∅ = bị xóa/thừa). Cần vì dấu rất nhỏ khó thấy bằng mắt."""
    d = [f"{e['r'] or '∅'}→{e['h'] or '∅'}" for e in rec["events"][:k]]
    return ", ".join(d) + (f" (+{len(rec['events']) - k})" if len(rec["events"]) > k else "")


def pick_examples(res, n=10, seed=0):
    """10 dự đoán đúng (ngẫu nhiên có seed) + 10 dự đoán sai chọn phân tầng theo loại lỗi
    (lần lượt từng loại phổ biến nhất, ưu tiên mẫu ít lỗi để dễ đọc). Không chọn theo ý chủ quan."""
    rng = random.Random(seed)
    recs = res["records"]
    ok = [r for r in recs if r["ref"] == r["hyp"]]
    correct = rng.sample(ok, min(n, len(ok)))
    wrong_pool = [r for r in recs if r["ref"] != r["hyp"]]
    order = [c for c, _ in sorted(res["cat"].items(), key=lambda kv: -kv[1])]
    chosen, used = [], set()
    while len(chosen) < min(n, len(wrong_pool)):
        progressed = False
        for c in order:
            cands = [r for r in wrong_pool if c in r["cats"] and id(r) not in used]
            if not cands:
                continue
            best = min(cands, key=lambda r: (r["edits"], rng.random()))
            chosen.append(best)
            used.add(id(best))
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return correct, chosen


# ====================================================================== khuyến nghị dựa trên số liệu
def recommendations(res) -> list[str]:
    if res["total_err"] == 0:
        return ["Không có lỗi nào ở tập này nên không có gì để đề xuất."]
    out, E = [], max(1, res["total_err"])
    share = lambda g: res["group"].get(g, 0) / E  # noqa: E731
    cer = res["cer"]
    if cer >= 0.4:
        ref, ci = res.get("ref_val_cer"), res["ci_cer"]
        head = f"CER = {100 * cer:.1f}% và Word Acc = {100 * res['word_acc']:.1f}% (khoảng tin cậy CER 95%: {100 * ci[0]:.1f}-{100 * ci[1]:.1f}%). "
        if ref is None:
            out.append(f"**CER của tập này rất cao.** {head}Chưa biết nguyên nhân là mô hình chưa hội tụ hay tập này khó/khác phân phối huấn luyện: "
                       f"hãy so với val CER trong `metrics.csv` (truyền `--val_cer` hoặc `--run` để công cụ tự so sánh). "
                       f"Tỉ lệ các loại lỗi bên dưới chỉ mang tính chỉ hướng.")
        elif ref < 0.5 * cer:
            out.append(f"**Ưu tiên 1: khoảng cách khái quát hóa, không phải thiếu huấn luyện.** {head}Cùng mô hình đạt val CER {100 * ref:.1f}% "
                       f"trên phân phối huấn luyện, tức là đã hội tụ ở đó nhưng suy giảm {cer / max(ref, 1e-9):.1f} lần ở tập này. "
                       f"Tập này khác phân phối (ví dụ font chưa thấy, nhiễu/mờ nặng) nên huấn luyện thêm trên cùng dữ liệu khó có thể giúp. "
                       f"Để tách nguyên nhân: sinh hai tập mới, một tập dùng font của train + augmentation nặng, một tập dùng font lạ + augmentation thường, "
                       f"rồi so CER; sau đó đưa một phần augmentation nặng và nhiều họ font hơn vào train.")
        else:
            out.append(f"**Ưu tiên 1: mô hình có thể chưa hội tụ.** {head}Val CER của run là {100 * ref:.1f}%, cùng mức với tập này. "
                       f"Việc cần làm trước: huấn luyện thêm (xem `val_cer` còn giảm không) rồi chạy lại phân tích; "
                       f"tỉ lệ các loại lỗi bên dưới chỉ mang tính chỉ hướng.")
    if share("deletion") >= 0.35:
        s = (f"**Mất ký tự chiếm {100 * share('deletion'):.1f}% số lỗi** (D = {res['D']} so với S = {res['S']}, I = {res['I']}). ")
        if res["mean_ratio_del"] is not None and res["mean_ratio_ok"] is not None:
            s += (f"Chỉ số T/(L+R) trung bình của mẫu có lỗi mất ký tự là {res['mean_ratio_del']:.2f}, "
                  f"so với {res['mean_ratio_ok']:.2f} ở mẫu đọc đúng hoàn toàn. ")
        s += "Thử `configs/model_s2.yaml` (stride 2) hoặc `model_h48.yaml` và so sánh số lỗi D trước/sau."
        out.append(s)
    dia = share("diacritic")
    if res["n_diac_chars"] and dia >= 0.2:
        tone = sum(res["cat"].get(c, 0) for c in ("tone_wrong", "tone_missing", "tone_extra"))
        mod = sum(res["cat"].get(c, 0) for c in ("modifier_wrong", "modifier_missing", "modifier_extra"))
        comp = res["cat"].get("diacritic_compound", 0)
        gap = res["ladder"]["đầy đủ"]["cer"] - res["ladder"]["bỏ dấu"]["cer"]
        worst = Counter()
        for p, n in res["pairs"].items():
            if p[0] != "∅" and has_diacritic(p[0]):
                worst[p[0]] += n
        top = ", ".join(f"'{c}' ({n})" for c, n in worst.most_common(5))
        out.append(f"**Lỗi dấu chiếm {100 * dia:.1f}% số lỗi** (dấu thanh {tone}, dấu phụ {mod}, cả hai {comp}). "
                   f"Bỏ dấu khỏi cả nhãn lẫn dự đoán làm CER giảm {100 * gap:.1f} điểm phần trăm "
                   f"({100 * res['ladder']['đầy đủ']['cer']:.1f}% -> {100 * res['ladder']['bỏ dấu']['cer']:.1f}%). "
                   f"Chữ có dấu bị lỗi nhiều nhất: {top}. Đề xuất: (1) tăng `img_height` lên 48 hoặc 64 để nét dấu rõ hơn "
                   f"(ở H=32 dấu chỉ còn khoảng 3 px); (2) sinh thêm dữ liệu có nhiều chữ hoa có dấu chồng và các chữ nêu trên; "
                   f"(3) dùng beam search kèm từ điển tiếng Việt để sửa dấu (chỉ có tác dụng nếu CER không dấu đã thấp).")
    look = share("lookalike")
    if look >= 0.1:
        top = [(p, n) for p, n in res["pairs"].most_common() if frozenset(p) in LOOKALIKE][:4]
        ts = ", ".join(f"{p[0]}->{p[1]} ({n})" for p, n in top)
        out.append(f"**Ký tự giống nhau chiếm {100 * look:.1f}% số lỗi** (nhiều nhất: {ts}). Với chuỗi mã/số không có ngữ nghĩa, "
                   f"mô hình ngôn ngữ không giúp được; hướng thử: thêm dữ liệu mã/số với nhiều font, tăng độ phân giải.")
    if share("case") >= 0.1:
        out.append(f"**Sai hoa/thường chiếm {100 * share('case'):.1f}% số lỗi.** CER không phân biệt hoa/thường là "
                   f"{100 * res['ladder']['bỏ hoa/thường']['cer']:.1f}% (so với {100 * cer:.1f}%). Nếu ứng dụng không cần phân biệt, báo cáo "
                   f"thêm chỉ số này; nếu cần, thêm dữ liệu cặp hoa/thường dễ nhầm (c/C, o/O, s/S, v/V, x/X, z/Z).")
    if share("space") >= 0.1:
        out.append(f"**Lỗi khoảng trắng chiếm {100 * share('space'):.1f}% số lỗi.** Xem xét đánh giá thêm theo từ, hoặc tăng "
                   f"tỉ lệ dòng nhiều từ trong dữ liệu.")
    bf = [r for r in res["by_font"] if r["n"] >= 20]
    if bf and bf[0]["cer"] >= 1.5 * max(cer, 1e-9):
        out.append(f"**Font `{bf[0]['key']}` có CER {100 * bf[0]['cer']:.1f}%** (n = {bf[0]['n']}), gấp {bf[0]['cer'] / max(cer, 1e-9):.1f} lần mức chung. "
                   f"Thử tăng tỉ lệ font kiểu này khi sinh dữ liệu hoặc đưa vào train thay vì chỉ để ở tập hard.")
    bl = res["by_length"]
    if len(bl) >= 2 and bl[0]["cer"] > 0 and bl[-1]["cer"] >= 1.5 * bl[0]["cer"] and bl[-1]["n"] >= 20:
        out.append(f"**CER tăng theo độ dài nhãn**: {100 * bl[0]['cer']:.1f}% ở nhóm {bl[0]['key']} ký tự lên {100 * bl[-1]['cer']:.1f}% ở nhóm "
                   f"{bl[-1]['key']} (n = {bl[-1]['n']}). Nên tăng tỉ lệ dòng dài trong huấn luyện hoặc kiểm tra lại T/(L+R) của các dòng dài.")
    bc = res["by_contrast"]
    if len(bc) >= 4 and bc[-1]["cer"] > 0 and bc[0]["cer"] >= 1.3 * bc[-1]["cer"]:
        out.append(f"**Ảnh tương phản thấp khó hơn rõ rệt**: CER {100 * bc[0]['cer']:.1f}% (25% ảnh tương phản thấp nhất) so với "
                   f"{100 * bc[-1]['cer']:.1f}% (25% cao nhất). Thử chuẩn hóa tương phản từng ảnh (vd. CLAHE) hoặc tăng augmentation tương phản.")
    if not out:
        out.append("Không có tín hiệu nào vượt ngưỡng quy tắc; hãy xem các bảng chi tiết bên dưới.")
    if res["total_err"] < 30:  # thêm SAU khi đã biết có quy tắc nào kích hoạt hay không
        out.append(f"(Chỉ có {res['total_err']} lỗi nên các tỉ lệ ở trên rất không ổn định.)")
    return out


# ====================================================================== vẽ hình
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"  # hiển thị được chữ có dấu và ký hiệu ∅
    return plt


def vis(c: str) -> str:
    return "␣" if c == " " else c


def fig_error_types(res, path, plt):
    items = sorted(res["cat"].items(), key=lambda kv: kv[1])
    colors = {"diacritic": "#d95f02", "lookalike": "#7570b3", "case": "#1b9e77", "space": "#e6ab02",
              "deletion": "#e7298a", "insertion": "#66a61e", "other": "#666666"}
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(items) + 1.2))
    ax.barh([CATS[c][0] for c, _ in items], [n for _, n in items], color=[colors[CATS[c][1]] for c, _ in items])
    for i, (_, n) in enumerate(items):
        ax.text(n, i, f" {n} ({100 * n / max(1, res['total_err']):.1f}%)", va="center", fontsize=8)
    ax.set(title=f"Phân bố lỗi theo loại (tổng {res['total_err']} lỗi)", xlabel="số lỗi")
    ax.set_xlim(0, max(n for _, n in items) * 1.25 if items else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_confusion(res, path, plt, k_rows=25):
    err_by_ref = Counter()
    for (r, h), n in res["pairs"].items():
        if r != "∅":
            err_by_ref[r] += n
    rows = [c for c, _ in err_by_ref.most_common(k_rows)]
    if not rows:
        return False
    cols, cell = [], {}
    for r in rows:
        row = Counter({h: n for (rr, h), n in res["pairs"].items() if rr == r})
        row[r] = res["equal_chars"].get(r, 0)
        for h, _ in row.most_common(4):
            if h not in cols:
                cols.append(h)
        cell[r] = row
    cols = sorted(cols, key=lambda c: (c == "∅", c))
    M = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        tot = sum(cell[r].values())
        for j, c in enumerate(cols):
            M[i, j] = cell[r].get(c, 0) / max(1, tot)
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(cols) + 2), 0.36 * len(rows) + 2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([vis(c) for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([vis(r) for r in rows], fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            n = cell[rows[i]].get(cols[j], 0)
            if n:
                ax.text(j, i, n, ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.5 else "black")
    ax.set(xlabel="ký tự dự đoán (∅ = bị xóa)", ylabel="ký tự nhãn",
           title=f"Ma trận nhầm lẫn: {len(rows)} ký tự nhãn lỗi nhiều nhất\n(màu = tỉ lệ theo hàng, số = số lần)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_bars(rows, title, xlabel, path, plt, overall=None):
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rows) + 2), 3.8))
    ax.bar([r["key"] for r in rows], [100 * r["cer"] for r in rows], color="#4c78a8")
    for i, r in enumerate(rows):
        ax.text(i, 100 * r["cer"], f"{100 * r['cer']:.1f}\nn={r['n']}", ha="center", va="bottom", fontsize=8)
    if overall is not None:
        ax.axhline(100 * overall, color="red", ls="--", lw=1, label=f"CER chung {100 * overall:.1f}%")
        ax.legend(fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel="CER (%)")
    ax.set_ylim(0, max(100 * r["cer"] for r in rows) * 1.3 + 1)
    plt.setp(ax.get_xticklabels(), rotation=20 if len(rows) > 6 else 0, ha="right" if len(rows) > 6 else "center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_tone(res, path, plt):
    if not res["tone_mat"]:
        return False
    M = np.zeros((6, 6))
    for (a, b), n in res["tone_mat"].items():
        M[TONE_ORDER.index(a), TONE_ORDER.index(b)] = n
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="Oranges")
    ax.set_xticks(range(6))
    ax.set_xticklabels(TONE_ORDER)
    ax.set_yticks(range(6))
    ax.set_yticklabels(TONE_ORDER)
    for i in range(6):
        for j in range(6):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9)
    ax.set(xlabel="dấu thanh dự đoán", ylabel="dấu thanh nhãn", title="Nhầm lẫn dấu thanh\n(cùng chữ gốc và dấu phụ)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_examples(items, title, path, data_dir, plt, with_cats=False):
    if not items:
        return False
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 1.15 * len(items) + 0.6))
    axes = np.atleast_1d(axes)
    from PIL import Image
    for ax, r in zip(axes, items):
        ax.axis("off")
        p = Path(data_dir) / r["path"] if data_dir else None
        if p is not None and p.exists():
            ax.imshow(np.asarray(Image.open(p).convert("L")), cmap="gray", aspect="equal")
        t = f"nhãn: {r['ref']}   |   dự đoán: {r['hyp']}"
        if with_cats:
            t += "   |   sai: " + diff_str(r) + "   |   " + ", ".join(CATS[c][0] for c in r["cats"])
        ax.set_title(t, fontsize=8.5, loc="left", pad=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# ====================================================================== báo cáo markdown
def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def write_report(res, out_dir: Path, data_dir, title, n_examples=10, seed=0):
    plt = setup_mpl()
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    E, N = max(1, res["total_err"]), res["n_chars"]
    L = [f"# Phân tích lỗi: {title}", ""]
    if res["cer"] >= 0.4:
        ref = res.get("ref_val_cer")
        if ref is not None and ref < 0.5 * res["cer"]:
            L += [f"> **Lưu ý:** CER tập này ({pct(res['cer'], 1)}%) cao gấp {res['cer'] / max(ref, 1e-9):.1f} lần val CER của cùng mô hình ({pct(ref, 1)}%): "
                  f"mô hình đã hội tụ trên phân phối huấn luyện, còn tập này khó hoặc khác phân phối.", ""]
        else:
            L += [f"> **Cảnh báo độ tin cậy:** CER = {pct(res['cer'], 1)}% rất cao" +
                  (" và chưa biết đây là do mô hình chưa hội tụ hay do tập khó (chưa có val CER để so)" if ref is None else f", cùng mức với val CER ({pct(ref, 1)}%) nên mô hình có thể chưa hội tụ") +
                  "; phần phân loại lỗi mang tính chỉ hướng.", ""]
    L += ["## 1. Tổng quan", "",
          md_table(["Chỉ số", "Giá trị"], [
              ["Số mẫu / số ký tự nhãn", f"{res['n']} / {res['n_chars']}"],
              ["CER", f"{pct(res['cer'])}% (95% CI: {pct(res['ci_cer'][0])}-{pct(res['ci_cer'][1])}%)"],
              ["Word Accuracy (khớp hoàn toàn)", f"{pct(res['word_acc'])}% (95% CI: {pct(res['ci_word'][0])}-{pct(res['ci_word'][1])}%)"],
              ["Thay thế S / Xóa D / Chèn I", f"{res['S']} / {res['D']} / {res['I']}  ({pct(res['S'] / max(1, N))}% / {pct(res['D'] / max(1, N))}% / {pct(res['I'] / max(1, N))}% số ký tự nhãn)"],
          ]), "", "Khoảng tin cậy bootstrap theo mẫu (1000 lần). CER = (S+D+I)/N gộp toàn tập.", ""]

    # --- 2. loại lỗi
    fig_error_types(res, figs / "error_types.png", plt)
    L += ["## 2. Phân bố lỗi theo loại", "", "![](figures/error_types.png)", "",
          md_table(["Loại lỗi", "Nhóm", "Số lỗi", "% tổng lỗi"],
                   [[CATS[c][0], GROUP_NAMES[CATS[c][1]], n, pct(n / E, 1)] for c, n in sorted(res["cat"].items(), key=lambda kv: -kv[1])]), "",
          "Theo nhóm lớn: " + "; ".join(f"{GROUP_NAMES[g]} {n} ({pct(n / E, 1)}%)" for g, n in sorted(res["group"].items(), key=lambda kv: -kv[1])), ""]

    # --- 3. cặp nhầm lẫn
    top = res["pairs"].most_common(20)
    rows = []
    for (r, h), n in top:
        rate = n / max(1, res["ref_chars"].get(r, 0)) if r != "∅" else None
        rows.append([f"`{vis(r)}` -> `{vis(h)}`", n, pct(n / E, 1), pct(rate, 1) if rate is not None else "—"])
    L += ["## 3. Ma trận nhầm lẫn và top-20 cặp nhầm lẫn", ""]
    if fig_confusion(res, figs / "confusion_matrix.png", plt):
        L += ["![](figures/confusion_matrix.png)", ""]
    L += ["Cặp ký tự (nhãn -> dự đoán; ∅ = bị xóa hoặc thừa):", "",
          md_table(["Cặp", "Số lần", "% tổng lỗi", "% lần xuất hiện của ký tự nhãn"], rows), "",
          "Cụm lỗi liền nhau (cho thấy cả các nhầm lẫn nhiều ký tự như m -> rn):", "",
          md_table(["Nhãn", "Dự đoán", "Số lần"], [[f"`{a or '∅'}`", f"`{b or '∅'}`", n] for (a, b), n in res["segs"].most_common(12)]), ""]

    # --- 4. ký tự giống nhau
    named = [("O", "0"), ("0", "O"), ("o", "0"), ("0", "o"), ("l", "1"), ("1", "l"), ("l", "I"), ("I", "l"), ("I", "1"), ("1", "I"),
             ("S", "5"), ("5", "S"), ("s", "5"), ("5", "s")]
    nrows = [[f"`{a}` -> `{b}`", res["pairs"].get((a, b), 0), res["ref_chars"].get(a, 0)] for a, b in named]
    seg_rows = [[f"`{a}` -> `{b}`", res["segs"].get((a, b), 0)] for a, b in sorted(DIGRAPHS)]
    L += ["## 4. Nhóm lỗi \"ký tự giống nhau\"", "",
          f"Nhóm này chiếm {res['group'].get('lookalike', 0)} lỗi ({pct(res['group'].get('lookalike', 0) / E, 1)}% tổng lỗi). "
          f"Nhóm đang xét (sửa được ở `LOOKALIKE_GROUPS`): {', '.join(LOOKALIKE_GROUPS)}.", "",
          md_table(["Cặp", "Số lần nhầm", "Số lần ký tự nhãn xuất hiện"], nrows), "",
          md_table(["Cặp nhiều ký tự", "Số lần"], seg_rows), ""]

    # --- 5. dấu tiếng Việt
    L += ["## 5. Lỗi dấu tiếng Việt", ""]
    if not res["n_diac_chars"]:
        L += ["Nhãn của tập này không có chữ nào mang dấu tiếng Việt nên bỏ qua phần này.", ""]
    else:
        c = res["cat"]
        g = lambda *k: sum(c.get(x, 0) for x in k)  # noqa: E731
        L += [f"Nhãn có {res['n_diac_chars']} ký tự mang dấu ({pct(res['n_diac_chars'] / max(1, N), 1)}% số ký tự). Phân loại bằng NFD:", "",
              md_table(["", "Sai loại", "Thiếu", "Thừa"], [
                  ["Dấu thanh (sắc/huyền/hỏi/ngã/nặng)", g("tone_wrong"), g("tone_missing"), g("tone_extra")],
                  ["Dấu phụ (mũ/móc/trăng/gạch đ)", g("modifier_wrong"), g("modifier_missing"), g("modifier_extra")]]),
              "", f"Sai cả hai loại dấu cùng lúc: {g('diacritic_compound')}; mất hẳn chữ có dấu: {g('del_vi_diacritic_letter')}.", ""]
        if fig_tone(res, figs / "tone_confusion.png", plt):
            L += ["![](figures/tone_confusion.png)", ""]
        if res["mod_detail"]:
            L += [md_table(["Loại", "Dấu phụ nhãn", "Dấu phụ dự đoán", "Số lần"],
                           [[CATS[k][0], a, b, n] for (k, a, b), n in res["mod_detail"].most_common(8)]), ""]
        lad = res["ladder"]
        full = lad["đầy đủ"]["cer"]
        L += ["### CER có dấu so với bỏ dấu", "",
              md_table(["Cách tính", "CER (%)", "Word Acc (%)", "Giảm so với đầy đủ (điểm %)"],
                       [[k, pct(v["cer"]), pct(v["word_acc"]), pct(full - v["cer"])] for k, v in lad.items()]), "",
              f"Bỏ dấu khỏi cả nhãn lẫn dự đoán làm CER giảm {pct(full - lad['bỏ dấu']['cer'])} điểm ({pct((full - lad['bỏ dấu']['cer']) / max(full, 1e-12), 1)}% của CER), "
              f"Word Acc tăng {pct(lad['bỏ dấu']['word_acc'] - lad['đầy đủ']['word_acc'])} điểm. Đây là **ước lượng** phần lỗi do dấu: bỏ dấu cũng làm thay đổi "
              f"các lỗi khác (vd. đ -> d), nên con số không tách bạch tuyệt đối.", ""]
        fig_bars([dict(key=k, cer=v["cer"], n=res["n"]) for k, v in lad.items()], "CER theo cách tính", "", figs / "cer_ladder.png", plt)
        L += ["![](figures/cer_ladder.png)", ""]

    # --- 6. lỗi theo đặc trưng
    L += ["## 6. Lỗi theo đặc trưng của mẫu", "", "Mỗi bảng ghi rõ số mẫu `n`; nhóm nhỏ (n < 30) có độ nhiễu cao.", ""]
    for name, key, ttl, xl in [("Theo độ dài nhãn (số ký tự)", "by_length", "CER theo độ dài nhãn", "độ dài nhãn"),
                               ("Theo kiểu chữ", "by_case", "CER theo kiểu chữ", "kiểu chữ"),
                               (f"Theo T/(L+R) (T = rộng@H={res['img_height']} chia {res['stride']}; càng thấp càng sát giới hạn CTC)", "by_ratio", "CER theo T/(L+R)", "T/(L+R)"),
                               ("Theo độ tương phản ảnh (độ lệch chuẩn điểm ảnh)", "by_contrast", "CER theo độ tương phản", "nhóm tương phản"),
                               ("Theo họ font (chỉ họ có đủ mẫu, 10 họ tệ nhất)", "by_font", "CER theo họ font", "họ font")]:
        rws = res[key][:10]
        if not rws:
            continue
        L += [f"### {name}", "", md_table(["Nhóm", "n", "CER (%)", "Word Acc (%)"], [[r["key"], r["n"], pct(r["cer"]), pct(r["word_acc"])] for r in rws]), ""]
        if fig_bars(rws, ttl, xl, figs / f"{key}.png", plt, overall=res["cer"]):
            L += [f"![](figures/{key}.png)", ""]

    # --- 7. ví dụ
    correct, wrong = pick_examples(res, n_examples, seed)
    for tag, items in (("correct", correct), ("wrong", wrong)):
        for i, r in enumerate(items, 1):
            src = Path(data_dir) / r["path"] if data_dir else None
            r["img"] = None
            if src is not None and src.exists():
                shutil.copy(src, ex_dir / f"{tag}_{i:02d}.png")
                r["img"] = f"examples/{tag}_{i:02d}.png"
    fig_examples(correct, f"{len(correct)} dự đoán ĐÚNG (chọn ngẫu nhiên, seed {seed})", figs / "examples_correct.png", data_dir, plt)
    fig_examples(wrong, f"{len(wrong)} dự đoán SAI (chọn phân tầng theo loại lỗi)", figs / "examples_wrong.png", data_dir, plt, with_cats=True)
    L += [f"## 7. Ví dụ dự đoán", "",
          "Dự đoán đúng được chọn ngẫu nhiên có seed; dự đoán sai được chọn lần lượt từ loại lỗi phổ biến nhất, ưu tiên mẫu ít lỗi cho dễ đọc "
          "(không chọn theo cảm tính). Cột nguyên nhân là **giả thuyết** suy từ loại lỗi và các đặc trưng đo được của ảnh, chưa kiểm chứng bằng thí nghiệm.", ""]
    if correct:
        L += ["### 10 dự đoán đúng", "", "![](figures/examples_correct.png)", "",
              md_table(["#", "Ảnh", "Nhãn = dự đoán", "Font"], [[i, f"![]({r['img']})" if r.get("img") else r["path"], f"`{r['ref']}`", r.get("font", "")] for i, r in enumerate(correct, 1)]), ""]
    if wrong:
        L += ["### 10 dự đoán sai", "", "![](figures/examples_wrong.png)", "",
              md_table(["#", "Ảnh", "Nhãn", "Dự đoán", "Chỗ sai", "Loại lỗi", "Nhận xét nguyên nhân"],
                       [[i, f"![]({r['img']})" if r.get("img") else r["path"], f"`{r['ref']}`", f"`{r['hyp']}`", f"`{diff_str(r)}`",
                         ", ".join(CATS[c][0] for c in r["cats"]), explain(r, res)] for i, r in enumerate(wrong, 1)]), ""]
    if not correct:
        L += ["(Không có dự đoán nào khớp hoàn toàn nên không có ví dụ đúng.)", ""]

    # --- 8. khuyến nghị
    L += ["## 8. Đề xuất cải thiện (rút ra từ số liệu ở trên)", "",
          "Mỗi đề xuất kèm con số quan sát được. Đây là tương quan từ một lần đánh giá, chưa phải bằng chứng nhân quả: "
          "cách kiểm chứng là thay đổi đúng một yếu tố và chạy lại `eval.py` + phân tích này.", ""]
    L += [f"{i}. {t}" for i, t in enumerate(recommendations(res), 1)]
    L += ["", "## 9. Phương pháp và giới hạn", "",
          "- Căn chỉnh bằng Levenshtein DP (khớp/thay > xóa > chèn khi hòa); căn chỉnh tối thiểu không duy nhất nên cặp nhầm lẫn phụ thuộc quy tắc hòa.",
          "- Mỗi thao tác lỗi thuộc đúng một loại; cụm lỗi khớp m<->rn, cl<->d... được gán 'cặp giống nhau' cho toàn cụm.",
          "- Chữ gốc và dấu tách bằng `unicodedata` NFD; 'đ' xử lý riêng (gạch ngang). Khi cả dấu phụ và dấu thanh cùng sai thì xếp vào loại 'sai cả hai'.",
          "- Họ font suy ra từ tên file; T/(L+R) tính với `--img_height` và `--stride` đã khai báo (mặc định 32 và 4).", ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    return correct, wrong


def save_tables(res, out_dir: Path):
    def w(name, head, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f, lineterminator="\n")
            cw.writerow(head)
            cw.writerows(rows)
    w("confusion_pairs.csv", ["ref", "hyp", "count"], [[r, h, n] for (r, h), n in res["pairs"].most_common()])
    w("error_types.csv", ["category", "group", "count", "share"],
      [[c, CATS[c][1], n, n / max(1, res["total_err"])] for c, n in sorted(res["cat"].items(), key=lambda kv: -kv[1])])
    for k in ("by_length", "by_case", "by_font", "by_ratio", "by_contrast"):
        w(f"{k}.csv", ["key", "n", "cer", "word_acc"], [[r["key"], r["n"], r["cer"], r["word_acc"]] for r in res[k]])
    summ = {k: res[k] for k in ("n", "n_chars", "S", "D", "I", "cer", "word_acc", "ci_cer", "ci_word", "total_err", "cat", "group", "ladder")}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")


# ====================================================================== đọc dữ liệu & CLI
def load_records(preds_path, data_dir=None, img_height=32, with_images=True):
    font = {}
    if data_dir and (Path(data_dir) / "labels.tsv").exists():
        for line in (Path(data_dir) / "labels.tsv").read_text(encoding="utf-8").split("\n"):
            p = line.split("\t")
            if len(p) >= 3:
                font[p[0]] = p[2]
    recs = []
    lines = Path(preds_path).read_text(encoding="utf-8").split("\n")
    for line in lines[1:]:
        if not line:
            continue
        p = line.split("\t")
        rec = dict(path=p[0], ref=unicodedata.normalize("NFC", p[1]), hyp=unicodedata.normalize("NFC", p[2] if len(p) > 2 else ""))
        rec["font"] = font.get(rec["path"])
        img = Path(data_dir) / rec["path"] if data_dir else None
        if with_images and img is not None and img.exists():
            from PIL import Image
            im = Image.open(img)
            w, h = im.size
            rec["width"] = min(768, max(16, round(w * img_height / h)))
            rec["contrast"] = float(np.asarray(im.convert("L"), dtype=np.float32).std())
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=None, help="file preds_<tập>_<decoder>.tsv do eval.py tạo")
    ap.add_argument("--tag", default=None, help="tên thư mục trong outputs/eval (thay cho --preds)")
    ap.add_argument("--set", default=None, help="tên tập, vd: vi_test, en_hard")
    ap.add_argument("--decoder", default="greedy")
    ap.add_argument("--data_dir", default=None, help="thư mục tập dữ liệu (có labels.tsv và images/); suy ra từ --set và --root nếu bỏ trống")
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--img_height", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4, help="tỉ lệ giảm chiều rộng của CNN (width_strides)")
    ap.add_argument("--n_examples", type=int, default=10)
    ap.add_argument("--min_font_n", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_cer", type=float, default=None, help="val CER của cùng mô hình (0-1) để phân biệt chưa hội tụ với khó khái quát")
    ap.add_argument("--run", default=None, help="tên run trong outputs/runs để lấy best_val_cer (mặc định: --tag nếu có)")
    a = ap.parse_args()

    if a.preds:
        preds = Path(a.preds)
    elif a.tag and a.set:
        preds = ROOT / "outputs" / "eval" / a.tag / f"preds_{a.set}_{a.decoder}.tsv"
    else:
        sys.exit("Cần --preds hoặc (--tag và --set)")
    if not preds.exists():
        sys.exit(f"Không thấy {preds}. Chạy trước: python -m src.eval --ckpt <run>")
    data_dir = a.data_dir
    if not data_dir:
        m = re.match(r"preds_([a-z]+)_([a-z]+)_", preds.name)
        if m:
            root = Path(a.root) if Path(a.root).is_absolute() else ROOT / a.root
            data_dir = root / m.group(1) / m.group(2)
    data_dir = Path(data_dir) if data_dir else None
    if data_dir is not None and not data_dir.exists():
        print(f"(không thấy {data_dir}: bỏ qua ảnh, font, tương phản)")
        data_dir = None
    out = Path(a.out) if a.out else ROOT / "outputs" / "analysis" / preds.parent.name / preds.stem.replace("preds_", "")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records(preds, data_dir, a.img_height)
    res = aggregate(recs, a.img_height, a.stride, a.bootstrap, a.seed, a.min_font_n)
    res["ref_val_cer"] = a.val_cer
    sj = ROOT / "outputs" / "runs" / (a.run or a.tag or preds.parent.name) / "summary.json"
    if res["ref_val_cer"] is None and sj.exists():
        res["ref_val_cer"] = json.loads(sj.read_text(encoding="utf-8")).get("best_val_cer")
    title = f"{preds.parent.name} / {preds.stem.replace('preds_', '')}"
    write_report(res, out, data_dir, title, a.n_examples, a.seed)
    save_tables(res, out)
    print(f"{res['n']} mẫu | CER {100 * res['cer']:.2f}% | Word Acc {100 * res['word_acc']:.2f}% | {res['total_err']} lỗi")
    for c, n in sorted(res["cat"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {CATS[c][0]:<48s} {n:>5d} ({100 * n / max(1, res['total_err']):.1f}%)")
    print(f"Báo cáo: {out / 'report.md'}")


if __name__ == "__main__":
    main()
