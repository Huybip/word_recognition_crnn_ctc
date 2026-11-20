"""Test Giai đoạn 6: demo.py, tiền xử lý dùng chung, báo cáo. Chạy: pytest -q tests/test_demo.py"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import predict  # noqa: E402
from src.dataset import OCRDataset, preprocess_image  # noqa: E402
from src.model import build_model  # noqa: E402
from src.text_utils import Charset  # noqa: E402

CKPT = ROOT / "checkpoints" / "mid_vi_best.pt"
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="chưa có checkpoints/mid_vi_best.pt")


def tiny_model():
    cs = Charset(list("abcdefgh "))
    mcfg = dict(img_height=32, cnn=dict(channels=[8, 16, 32, 64], width_strides=[2, 2]),
                rnn=dict(use_rnn=True, hidden=16, layers=1, dropout=0.0), head_dropout=0.0)
    torch.manual_seed(0)
    return build_model(cs.num_classes, mcfg).eval(), cs, mcfg


def make_png(path, w=120, h=40, mode="L"):
    im = Image.new(mode, (w, h), 255 if mode == "L" else (255, 255, 255, 255))
    ImageDraw.Draw(im).text((4, 12), "abc def", fill=20 if mode == "L" else (0, 0, 0, 255))
    im.save(path)
    return path


def test_preprocess_matches_dataset(tmp_path):
    p = make_png(tmp_path / "a.png", 150, 50)
    (tmp_path / "labels.tsv").write_text("a.png\tabc\tf.ttf\n", encoding="utf-8")
    ds = OCRDataset(tmp_path, Charset(list("abc ")), img_height=32)
    assert torch.equal(preprocess_image(p, 32)[0], ds.load_image(p))     # đúng như lúc huấn luyện
    assert preprocess_image(p, 48).shape[-2] == 48 and preprocess_image(p, 32).shape[:2] == (1, 1)


def test_preprocess_rgba_and_invert(tmp_path):
    im = Image.new("RGBA", (100, 40), (0, 0, 0, 0))                     # nền trong suốt
    ImageDraw.Draw(im).text((4, 12), "abc", fill=(0, 0, 0, 255))
    im.save(tmp_path / "t.png")
    x = preprocess_image(tmp_path / "t.png", 32)
    assert float(x.median()) > 0.9                                      # nền phải thành trắng, không phải đen
    p = make_png(tmp_path / "g.png")
    assert torch.allclose(preprocess_image(p, 32, invert=True), -preprocess_image(p, 32), atol=0.03)
    assert preprocess_image(Image.open(make_png(tmp_path / "r.png", mode="RGBA")), 32).shape[:2] == (1, 1)  # nhận cả PIL.Image


def test_predict_api(tmp_path):
    model, cs, mcfg = tiny_model()
    p = make_png(tmp_path / "a.png")
    g = predict(model, cs, mcfg, p, "greedy")
    assert len(g) == 1 and isinstance(g[0][0], str) and 0 < g[0][1] <= 1
    b = predict(model, cs, mcfg, p, "beam5", nbest=3)
    assert 1 <= len(b) <= 3 and all(b[i][1] >= b[i + 1][1] for i in range(len(b) - 1)) and sum(x[1] for x in b) <= 1 + 1e-6
    assert all(set(t) <= set(cs.chars) for t, _ in g + b)
    with pytest.raises(ValueError):
        predict(model, cs, mcfg, p, "viterbi")


def run_demo(*args):
    return subprocess.run([sys.executable, str(ROOT / "demo.py"), *args], capture_output=True, text=True, cwd=ROOT)


@needs_ckpt
def test_cli_output_format_and_errors(tmp_path):
    p = make_png(tmp_path / "x.png")
    r = run_demo("--ckpt", str(CKPT), str(p))
    assert r.returncode == 0 and r.stdout.count("\n") == 1 and r.stdout.split("\t")[0] == str(p)
    r = run_demo("--ckpt", str(CKPT), str(tmp_path / "khong_co.png"), str(p))   # ảnh hỏng không làm dừng cả lô
    assert r.returncode == 1 and "LỖI" in r.stderr and str(p) in r.stdout
    r = run_demo("--ckpt", str(CKPT), "--decoder", "beam3", "--nbest", "2", "--verbose", str(p))
    assert r.returncode == 0 and "#1" in r.stdout and "#2" in r.stdout
    assert run_demo("--ckpt", str(CKPT), str(tmp_path)).returncode == 0          # nhận cả thư mục


@needs_ckpt
def test_demo_equals_eval_at_batch_size_1():
    """demo xử lý từng ảnh riêng, nên phải trùng eval --batch_size 1 (không pad); khác eval batch 32 ở một số ảnh (xem báo cáo)."""
    f = ROOT / "outputs" / "eval" / "mid_vi_bs1" / "preds_vi_test_greedy.tsv"
    lab = ROOT / "samples" / "labels.tsv"
    if not (f.exists() and lab.exists()):
        pytest.skip("chưa có kết quả eval batch 1")
    ref = {l.split("\t")[0]: l.split("\t")[2] for l in f.read_text(encoding="utf-8").split("\n")[1:] if l}
    rows = [l.split("\t") for l in lab.read_text(encoding="utf-8").split("\n") if l]
    files = [str(ROOT / "samples" / r[0]) for r in rows]
    out = run_demo("--ckpt", str(CKPT), *files).stdout.strip().split("\n")
    assert len(out) == len(rows)
    for line, r in zip(out, rows):
        assert line.split("\t")[1] == ref[r[2]], r[0]


# ---------------------------------------------------------------- báo cáo
def test_report_numbers_come_from_outputs():
    import csv
    from src.build_report import build
    need = [ROOT / "outputs" / "runs" / "mid_vi" / "summary.json", ROOT / "outputs" / "eval" / "mid_vi" / "results.csv",
            ROOT / "outputs" / "eval" / "mid_vi_bs1" / "results.csv", ROOT / "outputs" / "stats" / "mid" / "vi_summary.csv"]
    if not all(p.exists() for p in need):
        pytest.skip("chưa có đủ outputs của mid_vi")
    md = build("mid_vi", ROOT)
    rows = {(r["set"], r["decoder"]): r for r in csv.DictReader(open(need[1], encoding="utf-8"))}
    assert f"{100 * float(rows[('vi_test', 'greedy')]['cer']):.2f}" in md and f"{100 * float(rows[('vi_hard', 'beam10')]['cer']):.2f}" in md
    assert "n/a" not in md and "{" not in md and "None" not in md
    assert 1200 < len(md.split()) < 3500
    for sec in ["## Tóm tắt", "## 1. Giới thiệu", "## 2. Phương pháp", "## 3. Thiết lập", "## 4. Kết quả", "## 5. Phân tích lỗi",
                "## 6. Hạn chế", "## 7. Hướng phát triển", "## 8. Tái lập"]:
        assert sec in md
    for kw in ["PARSeq", "TrOCR", "VietOCR", "mô hình ngôn ngữ", "chưa chạy"]:
        assert kw in md


# ---------------------------------------------------------------- notebook Colab
def run_notebook_cells(nb_path: Path, skip_prefixes=("!pip",)):
    """Chạy lần lượt các ô code của notebook trong một namespace: dòng '!lệnh' -> subprocess, bỏ '!pip'.
    (Chỉ kiểm tra MÃ của notebook chạy được ở môi trường cục bộ; KHÔNG phải chạy trên Colab thật.)"""
    import os
    import subprocess as sp

    import matplotlib
    import nbformat
    matplotlib.use("Agg")
    nb = nbformat.read(nb_path, as_version=4)
    ns, old = {"subprocess": sp}, os.getcwd()
    os.chdir(nb_path.parent)
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            lines = []
            for ln in cell.source.split("\n"):
                s, ind = ln.strip(), ln[:len(ln) - len(ln.lstrip())]
                if s.startswith("!"):
                    lines.append(ind + ("pass" if any(s.startswith(p) for p in skip_prefixes) else f"subprocess.run({s[1:]!r}, shell=True, check=True)"))
                else:
                    lines.append(ln)
            exec(compile("\n".join(lines), f"<ô {nb.cells.index(cell)}>", "exec"), ns)
    finally:
        os.chdir(old)
    return ns


@needs_ckpt
def test_notebook_cells_run_locally():
    pytest.importorskip("nbformat")
    nb = ROOT / "notebooks" / "colab_demo.ipynb"
    assert nb.exists()
    import nbformat
    nbformat.validate(nbformat.read(nb, as_version=4))
    ns = run_notebook_cells(nb)
    assert len(ns["hy_g"]) == len(ns["refs"]) == 12 and all(isinstance(t, str) for t in ns["hy_g"] + ns["hy_b"])
    assert ns["IN_COLAB"] is False and ns["n_par"] == 539_705
    out = run_demo("--ckpt", str(CKPT), *[str(ROOT / "samples" / f"vi_test_{i:02d}.png") for i in range(12)]).stdout.strip().split("\n")
    assert [l.split("\t")[1] for l in out] == ns["hy_g"]            # notebook và demo.py cho cùng kết quả
