"""Test Giai đoạn 3. Chạy: pytest -q tests/test_train.py"""
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ctc_decode import greedy_decode  # noqa: E402
from src.dataset import BucketBatchSampler, OCRDataset  # noqa: E402
from src.metrics import cer, cer_stats, levenshtein, levenshtein_py, word_accuracy  # noqa: E402
from src.model import CRNN, expand_fc  # noqa: E402
from src.text_utils import Charset  # noqa: E402
from src.train import build_scheduler, deep_update, load_config  # noqa: E402

DATA = ROOT / "data" / "synth"
CHARSET = ROOT / "data" / "charset.txt"
needs_data = pytest.mark.skipif(not (DATA / "en" / "train" / "labels.tsv").exists() or not CHARSET.exists(),
                                reason="chưa sinh dữ liệu (Giai đoạn 1)")


# ---------------------------------------------------------------- metrics
def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3 and levenshtein("abc", "") == 3
    assert levenshtein("Việt", "Viêt") == 1 and levenshtein("abc", "abc") == 0


def test_levenshtein_python_matches_rapidfuzz():
    rng = random.Random(0)
    for _ in range(300):
        a = "".join(rng.choices("abcdẹệ ", k=rng.randint(0, 12)))
        b = "".join(rng.choices("abcdẹệ ", k=rng.randint(0, 12)))
        assert levenshtein_py(a, b) == levenshtein(a, b)


def test_cer_is_pooled_not_mean_of_ratios():
    refs, hyps = ["a", "abcdefghi"], ["b", "abcdefghi"]  # lỗi 1 trên tổng 10 ký tự
    assert cer_stats(refs, hyps) == (1, 10) and cer(refs, hyps) == pytest.approx(0.1)
    assert word_accuracy(refs, hyps) == 0.5
    assert cer(["abc"], [""]) == 1.0 and cer(["abc"], ["abcd"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- greedy
def lp(seq, c=5):
    """log_probs (T,1,C) có đỉnh tại các index trong seq."""
    x = torch.full((len(seq), 1, c), -10.0)
    for t, k in enumerate(seq):
        x[t, 0, k] = 0.0
    return x


def test_greedy_collapse_and_blank():
    assert greedy_decode(lp([1, 1, 2, 2, 0, 3]))[0] == [1, 2, 3]
    assert greedy_decode(lp([1, 1, 1]))[0] == [1]                 # "aaa" liên tiếp -> "a"
    assert greedy_decode(lp([1, 0, 1]))[0] == [1, 1]              # blank tách ký tự lặp -> "aa"
    assert greedy_decode(lp([0, 0, 0]))[0] == []
    assert greedy_decode(lp([1, 2, 3, 4]), lengths=torch.tensor([2]))[0] == [1, 2]  # bỏ phần pad


def test_greedy_batch_and_charset():
    cs = Charset(["a", "b", "c", "d"])
    x = torch.cat([lp([1, 1, 0, 1]), lp([2, 3, 3, 0])], dim=1)
    assert [cs.decode(i) for i in greedy_decode(x)] == ["aa", "bc"]


# ---------------------------------------------------------------- cấu hình
def test_deep_update_and_config_inheritance(tmp_path):
    assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    (tmp_path / "base.yaml").write_text("x: {a: 1, b: 2}\ny: 3\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nx: {b: 20}\n")
    cfg = load_config(tmp_path / "child.yaml", ["x.a=5", "z=null", "w=[2,1]", "lr=3e-4"])
    assert cfg["x"] == {"a": 5, "b": 20} and cfg["y"] == 3 and cfg["z"] is None and cfg["w"] == [2, 1]
    assert cfg["lr"] == pytest.approx(3e-4)


def test_repo_configs_load():
    for n in ["small_cpu", "full_gpu", "full_gpu_vi_finetune", "full_gpu_vi_scratch"]:
        cfg = load_config(ROOT / "configs" / f"{n}.yaml")
        assert {"run", "data", "model", "optim", "sched", "train"} <= set(cfg)
    ft = load_config(ROOT / "configs" / "full_gpu_vi_finetune.yaml")
    assert ft["data"]["lang"] == "vi" and ft["init_from"] and ft["model"]["img_height"] == 48
    assert load_config(ROOT / "configs" / "full_gpu_vi_scratch.yaml")["init_from"] is None


# ---------------------------------------------------------------- scheduler
@pytest.mark.parametrize("sched", [dict(name="onecycle", pct_start=0.2), dict(name="cosine", warmup_steps=10, min_lr_ratio=0.01)])
def test_scheduler_shape(sched):
    p = nn.Linear(2, 2)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-3)
    s = build_scheduler(opt, sched, 1e-3, 100)
    lrs = []
    for _ in range(100):
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        s.step()
    assert max(lrs) == pytest.approx(1e-3, rel=0.05)
    assert lrs[0] < 0.2e-3 and lrs[-1] < 0.1e-3  # warmup rồi giảm về gần 0
    assert lrs.index(max(lrs)) < 40


# ---------------------------------------------------------------- mở rộng lớp Linear cuối
def test_expand_fc_keeps_old_weights_and_logits():
    torch.manual_seed(0)
    kw = dict(channels=(8, 16, 32, 64), rnn_hidden=32)
    old, new = CRNN(num_classes=7, **kw).eval(), CRNN(num_classes=20, **kw).eval()
    sd = old.state_dict()
    new.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    rep = expand_fc(new, sd["fc.weight"], sd["fc.bias"])
    assert rep == dict(n_old=7, n_new=20, n_added=13)
    assert torch.equal(new.fc.weight[:7], old.fc.weight) and torch.equal(new.fc.bias[:7], old.fc.bias)
    x = torch.randn(2, 1, 32, 96)
    with torch.no_grad():  # logits (trước softmax) của các lớp cũ giữ nguyên
        feat = lambda m: m.rnn(m.cnn(x).squeeze(2).permute(2, 0, 1))[0]  # noqa: E731
        assert torch.allclose(old.fc(feat(old)), new.fc(feat(new))[:, :, :7], atol=1e-6)
    assert new.fc.weight[7:].abs().sum() > 0


def test_expand_fc_rejects_shrinking():
    kw = dict(channels=(8, 16, 32, 64), rnn_hidden=32)
    small, big = CRNN(num_classes=5, **kw), CRNN(num_classes=9, **kw)
    with pytest.raises(ValueError):
        expand_fc(small, big.fc.weight, big.fc.bias)


# ---------------------------------------------------------------- sampler & dataset
def test_bucket_sampler_covers_all_and_groups_widths():
    rng = random.Random(1)
    widths = [rng.randint(20, 400) for _ in range(1000)]
    s = BucketBatchSampler(widths, 32, shuffle=True, seed=0)
    batches = list(s)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1000)) and len(batches) == len(s)
    spread = lambda bs: sum(max(widths[i] for i in b) - min(widths[i] for i in b) for b in bs) / len(bs)  # noqa: E731
    rand = [list(range(i, min(i + 32, 1000))) for i in range(0, 1000, 32)]
    assert spread(batches) < 0.5 * spread(rand)
    s.set_epoch(1)
    assert list(s) != batches  # epoch khác -> thứ tự khác


@needs_data
def test_dataset_cache_take_and_widths():
    cs = Charset.load(CHARSET)
    a = OCRDataset(DATA / "en" / "train", cs, img_height=32)
    b = OCRDataset(DATA / "en" / "train", cs, img_height=32, cache=True)
    assert torch.equal(a[3]["image"], b[3]["image"]) and a.widths() == b.widths()
    n = a.take(50, seed=1)
    assert len(n) == 50 and a.widths() == [a[i]["image"].shape[-1] for i in range(50)]
    b.take(50, seed=1)
    assert [s[1] for s in a.samples] == [s[1] for s in b.samples] and torch.equal(a[7]["image"], b[7]["image"])


def test_scientific_notation_in_yaml_file_is_float(tmp_path):
    (tmp_path / "c.yaml").write_text("optim: {lr: 1e-3, weight_decay: 1e-2}\n")  # YAML 1.1: thành chuỗi nếu không ép kiểu
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg["optim"]["lr"] == pytest.approx(1e-3) and cfg["optim"]["weight_decay"] == pytest.approx(1e-2)


def test_is_improved_ignores_noisy_cer_on_ctc_plateau():
    from src.train import is_improved
    first = dict(epoch=0, cer=float("inf"), loss=float("inf"))
    assert is_improved(dict(cer=1.0, loss=3.8), first)
    lucky = dict(epoch=1, cer=0.9157, loss=7.7)             # epoch 1 "may mắn": CER 91% nhưng loss rất tệ
    assert is_improved(dict(cer=1.0, loss=3.6), lucky)       # epoch sau chỉ-blank nhưng loss tốt hơn -> phải được chọn
    assert not is_improved(dict(cer=1.0, loss=4.0), dict(epoch=2, cer=1.0, loss=3.6))
    assert is_improved(dict(cer=0.50, loss=2.0), dict(epoch=3, cer=1.0, loss=3.0))          # thoát cao nguyên
    assert is_improved(dict(cer=0.30, loss=1.0), dict(epoch=4, cer=0.50, loss=0.9))         # dưới cao nguyên: CER quyết định
    assert not is_improved(dict(cer=0.499, loss=0.5), dict(epoch=4, cer=0.50, loss=0.9), min_delta=0.01)
    assert is_improved(dict(cer=0.5, loss=0.8), dict(epoch=4, cer=0.5, loss=0.9))           # hòa CER -> loss quyết định
