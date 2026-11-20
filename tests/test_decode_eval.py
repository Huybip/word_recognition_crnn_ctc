"""Test Giai đoạn 4: CER, greedy, prefix beam search, mã hóa NFC, eval. Chạy: pytest -q tests/test_decode_eval.py"""
import itertools
import math
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ctc_decode import beam_decode, greedy_decode, prefix_beam_search  # noqa: E402
from src.metrics import align, cer, corpus_metrics, edit_counts, levenshtein, levenshtein_py  # noqa: E402
from src.text_utils import Charset  # noqa: E402

A, B, C_ = 1, 2, 3  # ba "ký tự" trong các ca thử; 0 = blank


def onehot(seq, c=5, p=0.999):
    """log-probs (T,1,C) cực nhọn: bước t gần như chắc chắn là seq[t]."""
    x = torch.full((len(seq), 1, c), math.log((1 - p) / (c - 1)))
    for t, k in enumerate(seq):
        x[t, 0, k] = math.log(p)
    return x


# ============================================================ CER
@pytest.mark.parametrize("ref,hyp,sdi", [
    ("abc", "abc", (0, 0, 0)),
    ("abc", "abd", (1, 0, 0)),      # thay
    ("abc", "ab", (0, 1, 0)),       # xóa (mất ký tự)
    ("abc", "abxc", (0, 0, 1)),     # chèn (thừa ký tự)
    ("kitten", "sitting", (2, 0, 1)),
    ("abc", "", (0, 3, 0)),         # dự đoán rỗng
    ("", "ab", (0, 0, 2)),          # nhãn rỗng
    ("", "", (0, 0, 0)),
    ("Việt", "Viêt", (1, 0, 0)),    # sai dấu = 1 lỗi thay
])
def test_edit_counts_cases(ref, hyp, sdi):
    assert edit_counts(ref, hyp) == sdi
    assert sum(sdi) == levenshtein(ref, hyp)


def test_cer_values_and_pooling():
    assert cer(["abc"], ["abc"]) == 0 and cer(["abc"], [""]) == 1.0 and cer(["abc"], ["abcd"]) == pytest.approx(1 / 3)
    m = corpus_metrics(["a", "abcdefghi"], ["b", "abcdefghi"])  # 1 lỗi / 10 ký tự, KHÔNG phải trung bình (1.0 + 0)/2
    assert m["cer"] == pytest.approx(0.1) and m["word_acc"] == 0.5 and (m["S"], m["D"], m["I"]) == (1, 0, 0)
    assert corpus_metrics([], [])["cer"] == 0  # không chia cho 0
    assert corpus_metrics([""], [""])["word_acc"] == 1.0


def test_cer_random_invariants_and_python_fallback():
    rng = random.Random(0)
    for _ in range(300):
        r = "".join(rng.choices("abcdẹệ ", k=rng.randint(0, 12)))
        h = "".join(rng.choices("abcdẹệ ", k=rng.randint(0, 12)))
        s, d, i = edit_counts(r, h)
        assert s + d + i == levenshtein(r, h) == levenshtein_py(r, h)
        ops = align(r, h)
        assert sum(o[0] != "equal" for o in ops) == s + d + i          # căn chỉnh tối thiểu
        rebuilt = "".join(h[j] if o in ("equal", "sub", "ins") else "" for o, _i, j in ops)
        assert rebuilt == h                                              # áp dụng ops phải ra đúng dự đoán
        assert sum(o in ("equal", "sub", "del") for o, _i, _j in ops) == len(r)  # mọi ký tự nhãn đều xuất hiện


def test_cer_matches_jiwer():
    jiwer = pytest.importorskip("jiwer")
    rng = random.Random(1)
    refs, hyps = [], []
    for _ in range(200):
        r = "".join(rng.choices("abcdeẹệư", k=rng.randint(1, 10)))  # không rỗng, không khoảng trắng đầu/cuối/kép (jiwer sẽ chuẩn hóa)
        h = "".join(rng.choices("abcdeẹệư", k=rng.randint(0, 10)))
        refs.append(r)
        hyps.append(h)
    assert corpus_metrics(refs, hyps)["cer"] == pytest.approx(jiwer.cer(refs, hyps))


# ============================================================ greedy
def test_greedy_repeats_blank_separator():
    assert greedy_decode(onehot([A, A]))[0] == [A]            # "aa" liền kề  -> "a"
    assert greedy_decode(onehot([A, 0, A]))[0] == [A, A]      # "a-a" -> "aa"
    assert greedy_decode(onehot([A, A, 0, A, A]))[0] == [A, A]
    assert greedy_decode(onehot([0, 0]))[0] == [] and greedy_decode(onehot([A, B, B, C_]))[0] == [A, B, C_]


# ============================================================ prefix beam search
def test_beam_blank_and_repeat_semantics():
    assert beam_decode(onehot([A, A]), beam_width=5)[0] == [A]
    assert beam_decode(onehot([A, 0, A]), beam_width=5)[0] == [A, A]
    assert beam_decode(onehot([0, 0, 0]), beam_width=5)[0] == []
    assert beam_decode(onehot([A, A, A, 0, B, B]), beam_width=5)[0] == [A, B]


def test_beam1_equals_greedy_on_peaked_outputs():
    """beam=1 KHÔNG luôn trùng greedy (xem test kế tiếp); nhưng với đầu ra nhọn thì phải trùng."""
    rng = random.Random(0)
    for _ in range(200):
        seq = [rng.randint(0, 4) for _ in range(rng.randint(1, 20))]
        x = onehot(seq)
        assert beam_decode(x, beam_width=1)[0] == greedy_decode(x)[0], seq


def test_beam1_can_differ_from_greedy_when_flat():
    """Ví dụ phản chứng: bước 2 phân phối phẳng. Greedy chọn 'b' (0.34 là lớn nhất) -> "ab",
    nhưng tổng xác suất của "a" (blank 0.33 + lặp 0.33) = 0.6*0.66 > "ab" = 0.6*0.34."""
    p = torch.tensor([[[0.01, 0.98, 0.01]], [[0.33, 0.33, 0.34]]])  # (T=2,N=1,C=3)
    lp = p.log()
    assert greedy_decode(lp)[0] == [1, 2]
    assert beam_decode(lp, beam_width=1)[0] == [1]


@pytest.mark.parametrize("seed", range(6))
def test_large_beam_is_exact_map(seed):
    """Đối chiếu brute-force: liệt kê MỌI alignment (C^T đường), cộng xác suất theo chuỗi rút gọn.
    Beam đủ lớn, không cắt tỉa -> phải ra đúng chuỗi xác suất cao nhất và đúng giá trị xác suất."""
    torch.manual_seed(seed)
    T, C = 5, 4
    lp = torch.log_softmax(torch.randn(T, C) * 1.5, dim=1)
    prob = {}
    for path in itertools.product(range(C), repeat=T):
        out, prev = [], None
        for c in path:
            if c != prev and c != 0:
                out.append(c)
            prev = c
        prob[tuple(out)] = prob.get(tuple(out), 0.0) + math.exp(sum(lp[t, c].item() for t, c in enumerate(path)))
    best = max(prob.items(), key=lambda kv: kv[1])
    ids, score = prefix_beam_search(lp.numpy(), beam_width=10**6, prune=float("-inf"))[0]
    assert tuple(ids) == best[0] and math.exp(score) == pytest.approx(best[1], rel=1e-9)
    # Mọi prefix trong beam phải có xác suất đúng bằng tổng brute-force của chính nó
    for ids2, s2 in prefix_beam_search(lp.numpy(), beam_width=10**6, prune=float("-inf")):
        assert math.exp(s2) == pytest.approx(prob.get(tuple(ids2), 0.0), rel=1e-9)


def test_beam_sorted_and_width_respected():
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(8, 6), dim=1).numpy()
    res = prefix_beam_search(lp, beam_width=4)
    assert len(res) <= 4 and all(res[i][1] >= res[i + 1][1] for i in range(len(res) - 1))


def test_beam_respects_lengths_and_batch():
    x = torch.cat([onehot([A, B, C_, C_]), onehot([B, 0, B, A])], dim=1)
    assert beam_decode(x, torch.tensor([2, 4]), beam_width=5) == [[A, B], [B, B, A]]


def test_beam_matches_pyctcdecode_on_peaked():
    pytest.importorskip("pyctcdecode")
    from src.ctc_decode import make_pyctcdecode
    cs = ["a", "b", "c", "d"]
    run = make_pyctcdecode(cs, beam_width=10)
    rng = random.Random(0)
    for _ in range(30):
        seq = [rng.randint(0, 4) for _ in range(rng.randint(2, 15))]
        x = onehot(seq)
        assert run(x)[0] == beam_decode(x, beam_width=10)[0] == greedy_decode(x)[0], seq


# ============================================================ mã hóa/giải mã nhãn NFC
@pytest.mark.parametrize("s", ["Việt Nam", "ĐỔI MỚI", "thuở ấy ữ ặ", "Ngọc 12/05/2024", "ẤẦẨẪẬ"])
def test_encode_decode_nfc(s):
    cs = Charset(sorted(set(unicodedata.normalize("NFC", s))))
    nfd = unicodedata.normalize("NFD", s)
    assert cs.encode(nfd) == cs.encode(s)                       # NFD đầu vào được chuẩn hóa về NFC
    out = cs.decode(cs.encode(s))
    assert out == unicodedata.normalize("NFC", s) and unicodedata.is_normalized("NFC", out)


def test_encode_non_strict_and_unknown():
    cs = Charset(["a", "b"])
    assert cs.encode("abz", strict=False) == [1, 2]
    with pytest.raises(KeyError):
        cs.encode("abz")


# ============================================================ eval end-to-end (mô hình nhỏ, dữ liệu smoke nếu có)
DATA = ROOT / "data" / "synth"
needs_data = pytest.mark.skipif(not (DATA / "vi" / "test" / "labels.tsv").exists() or not (ROOT / "data" / "charset.txt").exists(),
                                reason="chưa sinh dữ liệu")


@needs_data
def test_eval_pipeline_end_to_end(tmp_path):
    from src.eval import default_sets, evaluate_sets, load_checkpoint, save_outputs
    from src.model import build_model
    cs = Charset.load(ROOT / "data" / "charset_en.txt")  # charset EN: nhãn VI sẽ có OOV
    mcfg = dict(img_height=32, cnn=dict(channels=[8, 16, 32, 64], width_strides=[2, 2]),
                rnn=dict(use_rnn=True, hidden=16, layers=1, dropout=0.0), head_dropout=0.0)
    torch.manual_seed(0)
    model = build_model(cs.num_classes, mcfg)
    torch.save(dict(model=model.state_dict(), charset=cs.chars, model_cfg=mcfg, epoch=0, val={}), tmp_path / "m.pt")
    m2, cs2, mcfg2, _ = load_checkpoint(tmp_path / "m.pt", torch.device("cpu"))
    sets = default_sets(DATA, ["en/test", "vi/test"])
    rows, preds = evaluate_sets(m2, cs2, mcfg2, sets, ["greedy", "beam3"], batch_size=8, max_samples=16)
    assert len(rows) == 4 and {r["decoder"] for r in rows} == {"greedy", "beam3"}
    for r in rows:
        assert r["n"] == 16 and r["cer"] >= 0 and 0 <= r["word_acc"] <= 1 and r["ms_total"] >= r["ms_decode"] >= 0
        assert r["cer"] == pytest.approx(r["sub"] + r["dele"] + r["ins"])
    assert next(r for r in rows if r["set"] == "vi_test")["oov_rate"] > 0      # nhãn VI có ký tự ngoài charset EN
    assert next(r for r in rows if r["set"] == "en_test")["oov_rate"] == 0
    ref_hyp = preds[("en_test", "greedy")]
    assert len(ref_hyp) == 16 and all(isinstance(h, str) for _, _, h in ref_hyp)
    save_outputs(tmp_path / "out", rows, preds, "t")
    assert (tmp_path / "out" / "results.md").read_text(encoding="utf-8").count("| en_test |") >= 2
    assert (tmp_path / "out" / "preds_vi_test_beam3.tsv").exists()


# ============================================================ ablation (lập kế hoạch + dựng bảng, không train thật)
def test_ablation_plan_and_commands(tmp_path):
    from src.ablation import plan, train_cmd
    for sp, n in [("train", 100), ("train_clean", 20)]:
        d = tmp_path / "en" / sp
        d.mkdir(parents=True)
        (d / "labels.tsv").write_text("".join(f"images/{i}.png\ttext\tf.ttf\n" for i in range(n)))
    ex = plan("t", tmp_path, 1e-3, {"scratch_vi": "vi_scratch"}, None)
    assert set(ex) == {"base_en", "norn_en", "aug_en", "noaug_en", "ft_vi", "scratch_vi"}
    assert ex["aug_en"]["overrides"] == {"data.max_train_samples": 20}          # cùng số mẫu với train_clean
    assert ex["noaug_en"]["overrides"]["data.train_split"] == "train_clean"
    assert ex["ft_vi"]["overrides"]["init_from"].endswith("ab_t_base_en/best.pt") and ex["ft_vi"]["overrides"]["optim.lr"] == 1e-3
    assert ex["scratch_vi"]["run"] == "vi_scratch" and ex["scratch_vi"]["overrides"]["init_from"] is None
    cmd = train_cmd(Path("configs/x.yaml"), ex["norn_en"], 3)
    assert "model.rnn.use_rnn=false" in cmd and "train.epochs=3" in cmd and "run.name=ab_t_norn_en" in cmd and "init_from=null" not in cmd
    assert "init_from=null" in train_cmd(Path("c.yaml"), ex["scratch_vi"], None)
    (tmp_path / "en" / "train_clean" / "labels.tsv").write_text("".join(f"images/{i}.png\ttext\tf.ttf\n" for i in range(100)))
    assert plan("t", tmp_path, None, {}, None)["aug_en"]["alias_of"] == "base_en"   # bằng số mẫu -> dùng lại base
    assert "noaug_en" not in plan("t", tmp_path / "none", None, {}, None)           # thiếu train_clean -> bỏ qua


def test_ablation_summary_tables():
    from src.ablation import build_summary
    row = lambda s, d, cer: dict(set=s, decoder=d, cer=cer, word_acc=0.5, ms_total=3.0, ms_decode=1.0)  # noqa: E731
    res = {"base_en": dict(rows=[row("en_test", "greedy", 0.1), row("en_test", "beam5", 0.09), row("en_hard", "greedy", 0.3)],
                           summary=dict(params=2_000_000, best_epoch=7, best_val_cer=0.08)),
           "norn_en": dict(rows=[row("en_test", "greedy", 0.2)], summary=dict(params=900_000)),
           "ft_vi": dict(rows=[row("vi_test", "greedy", 0.15)], summary=dict(best_epoch=3, best_val_cer=0.12))}
    md = build_summary(res, ["greedy", "beam5"])
    for h in ["Ablation 1", "Ablation 2", "Ablation 3", "Ablation 4"]:
        assert h in md
    assert "| CNN + BiLSTM + Linear | 2,000,000 | 10.00 | 50.00 | 30.00 |" in md
    assert "| CNN + Linear (không RNN) | 900,000 | 20.00 |" in md
    assert "| en_test | beam5 | 9.00 |" in md and "| fine-tune từ EN | 15.00 |" in md and "| từ đầu | — |" in md  # thiếu số liệu -> "—"
