#!/usr/bin/env python
"""Sinh REPORT.md từ các file trong outputs/. MỌI con số trong báo cáo được đọc từ outputs/ (không gõ tay),
nên chạy lại script sau khi huấn luyện/đánh giá lại sẽ cập nhật toàn bộ báo cáo.

  python -m src.build_report [--run mid_vi] [--out REPORT.md]
Cần có: outputs/runs/<run>/{summary.json,metrics.csv,config_resolved.yaml}, outputs/eval/<run>/results.csv,
        outputs/analysis/<run>/{vi_test_greedy,vi_hard_greedy}/, outputs/stats/mid/vi_summary.csv, outputs/checks/model_h{32,48}.txt,
        outputs/runs/overfit200_{en,vi_ft,vi_scratch}/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analyze_errors import CATS, GROUP_NAMES  # noqa: E402


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"Thiếu {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}. Chạy các bước tương ứng trước (xem README).")
    return p


def rcsv(p: Path):
    with open(need(p), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rjson(p: Path):
    return json.loads(need(p).read_text(encoding="utf-8"))


def pc(x, nd=2):
    return f"{100 * float(x):.{nd}f}"


def first_epoch(rows, thr):
    return next((int(r["epoch"]) for r in rows if float(r["val_cer"]) < thr), None)


def parse_check(txt: str) -> dict:
    d = {}
    m = re.search(r"Tham số: tổng ([\d,]+)", txt)
    d["params"] = m.group(1) if m else "n/a"
    m = re.search(r"Số mẫu: (\d+) \| nhãn dài nhất Lmax = (\d+)", txt)
    d["n"], d["lmax"] = (m.group(1), m.group(2)) if m else ("n/a", "n/a")
    for s in (4, 2):
        m = re.search(rf"stride {s}: T\(rộng max\)=\s*(\d+) .*?CHÍNH XÁC \(T<L\+R\):\s*(\d+) \| vi phạm bảo thủ \(T<2L\+1\):\s*(\d+)", txt)
        d[s] = m.groups() if m else ("n/a",) * 3
    return d


def build(run: str, out_dir: Path) -> str:
    O = ROOT / "outputs"
    summ = rjson(O / "runs" / run / "summary.json")
    cfg = yaml.safe_load(need(O / "runs" / run / "config_resolved.yaml").read_text(encoding="utf-8"))
    met = rcsv(O / "runs" / run / "metrics.csv")
    ev = {(r["set"], r["decoder"]): r for r in rcsv(O / "eval" / run / "results.csv")}
    ev1 = {(r["set"], r["decoder"]): r for r in rcsv(O / "eval" / f"{run}_bs1" / "results.csv")}
    st = rcsv(O / "stats" / "mid" / "vi_summary.csv")
    chk = {h: parse_check(need(O / "checks" / f"model_h{h}.txt").read_text(encoding="utf-8")) for h in (32, 48)}
    an = {s: rjson(O / "analysis" / run / f"{s}_greedy" / "summary.json") for s in ("vi_test", "vi_hard")}
    bycon = {s: rcsv(O / "analysis" / run / f"{s}_greedy" / "by_contrast.csv") for s in an}
    bylen = rcsv(O / "analysis" / run / "vi_test_greedy" / "by_length.csv")
    pairs = rcsv(O / "analysis" / run / "vi_test_greedy" / "confusion_pairs.csv")[:6]
    over = {k: (rjson(O / "runs" / k / "summary.json"), rcsv(O / "runs" / k / "metrics.csv"))
            for k in ("overfit200_en", "overfit200_vi_ft", "overfit200_vi_scratch")}

    def diff_count(sname):
        def rd(tag):
            f = O / "eval" / tag / f"preds_{sname}_greedy.tsv"
            return {l.split("\t")[0]: l.split("\t")[2] for l in need(f).read_text(encoding="utf-8").split("\n")[1:] if l}
        a, b = rd(run), rd(f"{run}_bs1")
        return sum(a[k] != b[k] for k in a), len(a)

    d_test, n_test = diff_count("vi_test")
    d_hard, n_hard = diff_count("vi_hard")
    m, o = cfg["model"], cfg["optim"]
    tot_min = sum(float(r["sec"]) for r in met) / 60
    sec_ep = sum(float(r["sec"]) for r in met) / len(met)
    leave = first_epoch(met, 0.5)
    last = met[-1]
    T = an["vi_test"]
    E_t, E_h = max(1, T["total_err"]), max(1, an["vi_hard"]["total_err"])
    g = lambda a, k: a["group"].get(k, 0)  # noqa: E731
    tone = sum(T["cat"].get(c, 0) for c in ("tone_wrong", "tone_missing", "tone_extra"))
    modi = sum(T["cat"].get(c, 0) for c in ("modifier_wrong", "modifier_missing", "modifier_extra"))
    lad = T["ladder"]
    c_test = {d: ev[("vi_test", d)] for d in ("greedy", "beam10")}
    c_hard = {d: ev[("vi_hard", d)] for d in ("greedy", "beam10")}

    def row_eval(setn, d, r):
        return f"| {setn} | {d} | {pc(r['cer'])} | {pc(r['word_acc'])} | {float(r['ms_total']):.2f} | {float(r['ms_decode']):.2f} |"

    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p  # noqa: E731
    stats_rows = "\n".join(f"| {r['tập']} | {r['số mẫu']} | {r['nhãn khác nhau']} | {r['len TB']} | {r['len max']} |" for r in st)
    over_rows = "\n".join(
        f"| {name} | {first_epoch(rows, 0.05) or 'chưa đạt'} | {pc(rows[-1]['val_cer'])} | {pc(rows[-1]['val_word_acc'], 1)} | {len(rows)} |"
        for name, (sm, rows) in over.items())
    grp = "\n".join(f"| {GROUP_NAMES[k]} | {g(T, k)} | {pc(g(T, k) / E_t, 1)} | {g(an['vi_hard'], k)} | {pc(g(an['vi_hard'], k) / E_h, 1)} |"
                    for k in ("diacritic", "other", "deletion", "case", "space", "insertion", "lookalike"))
    con = lambda s, i: pc(float(bycon[s][i]["cer"]), 1)  # noqa: E731
    top_pairs = "; ".join(f"`{p['ref']}`→`{p['hyp']}` ({p['count']})" for p in pairs[:5])
    len_rows = "; ".join(f"{r['key']} ký tự: {pc(r['cer'], 1)}%" for r in bylen)
    hard_word = pc(c_hard["greedy"]["word_acc"], 1)

    md = f"""# Nhận dạng chữ trong ảnh bằng CRNN + CTC (tiếng Anh và tiếng Việt có dấu)

> Mọi số liệu trong tài liệu này được `src/build_report.py` đọc từ thư mục `outputs/`. Toàn bộ thực nghiệm chạy trên **dữ liệu sinh (TRDG)**,
> trên **CPU 1 core**, với **một mô hình nhỏ** và **một lần chạy (một seed)**. Chưa có thực nghiệm trên ảnh thật hay trên GPU.

## Tóm tắt

Dự án cài đặt và đánh giá mô hình CRNN (CNN + BiLSTM) huấn luyện bằng CTC để đọc một dòng chữ trong ảnh, có phần thử riêng cho tiếng Việt có dấu.
Mô hình nhỏ `{run}` ({summ['params']:,} tham số, {summ['epochs_run']} epoch, {tot_min:.0f} phút trên CPU) đạt **CER {pc(c_test['greedy']['cer'])}%** và
**Word Accuracy {pc(c_test['greedy']['word_acc'])}%** trên tập test tiếng Việt ({ev[('vi_test', 'greedy')]['n']} mẫu, cùng phân phối với tập huấn luyện), nhưng chỉ đạt CER {pc(c_hard['greedy']['cer'])}%
trên tập hard (font chưa từng thấy và nhiễu/mờ nặng). Lỗi dấu chiếm {pc(g(T, 'diacritic') / E_t, 1)}% số lỗi trên tập test; bỏ dấu khỏi nhãn và dự đoán làm CER giảm từ
{pc(lad['đầy đủ']['cer'])}% xuống {pc(lad['bỏ dấu']['cer'])}%. Beam search (beam 10) chỉ cải thiện CER {pc(float(c_test['greedy']['cer']) - float(c_test['beam10']['cer']))} điểm trên tập test,
nằm trong khoảng nhiễu. Các ablation (BiLSTM, augmentation, fine-tune EN→VI) chỉ được kiểm tra ở mức pipeline chạy thông, **chưa có kết quả có ý nghĩa**.

## 1. Giới thiệu

Bài toán là nhận dạng dòng chữ trong ảnh (line-level text recognition). Cách làm kinh điển cho một dòng chữ có độ dài thay đổi là CRNN + CTC:
CNN trích đặc trưng, RNN hai chiều mô hình hóa chuỗi theo chiều ngang, và hàm mất mát CTC cho phép huấn luyện mà **không cần căn chỉnh từng ký tự** với ảnh.
Mục tiêu của dự án: (1) chuẩn bị dữ liệu ảnh chữ, (2) cài đặt CNN + BiLSTM + CTC, (3) giải mã bằng greedy và beam search, (4) đánh giá bằng CER và Word Accuracy,
(5) thử trên chữ tiếng Việt có dấu và (6) phân tích lỗi (ký tự giống nhau, dấu).

## 2. Phương pháp

**Kiến trúc.** CNN kiểu VGG nhỏ gồm 6 lớp Conv3×3 + BatchNorm + ReLU xen MaxPool, sau đó một Conv có kernel cao H/16 đưa chiều cao về 1, giữ chiều rộng làm
trục thời gian với T = W/4. Tiếp theo là 2 lớp BiLSTM (hidden 256), lớp Linear sang số lớp = số ký tự + 1 (blank ở index 0) và log-softmax. Cấu hình đầy đủ có
{chk[32]['params']} tham số; mô hình đã huấn luyện (`{run}`) dùng kênh {m['cnn']['channels']} và hidden {m['rnn']['hidden']} để chạy được trên CPU.

**CTC và điều kiện về độ dài.** CTC chỉ căn chỉnh được khi T ≥ L + R (L là độ dài nhãn, R là số cặp ký tự giống nhau liền kề, vì giữa hai ký tự lặp cần một blank).
Kiểm tra trên {chk[32]['n']} mẫu (bộ smoke, nhãn dài nhất {chk[32]['lmax']} ký tự): ở H=32, stride 4 có {chk[32][4][1]} mẫu vi phạm điều kiện chính xác nhưng {chk[32][4][2]} mẫu dưới ngưỡng bảo thủ T ≥ 2L+1
(tập trung ở font hẹp); ở H=48, stride 4 chỉ còn {chk[48][4][2]} mẫu. `zero_infinity=True` sẽ âm thầm bỏ qua mẫu vô nghiệm nên mã huấn luyện đếm và ghi lại tỉ lệ này.

**Giải mã.** Greedy (argmax từng bước, gộp ký tự lặp, bỏ blank) và prefix beam search tự cài, tính trong không gian log, cộng xác suất mọi alignment của cùng một chuỗi.
Cài đặt được đối chiếu với liệt kê toàn bộ alignment (brute-force) trên các ca nhỏ: beam đủ lớn cho đúng chuỗi và đúng xác suất. Beam=1 **không** luôn trùng greedy.

**Chỉ số.** CER = (S + D + I) / N gộp toàn tập (N là tổng số ký tự nhãn), Word Accuracy là tỉ lệ dòng khớp hoàn toàn nhãn. Khoảng tin cậy 95% dùng bootstrap theo mẫu (1000 lần).

## 3. Thiết lập thí nghiệm

**Dữ liệu.** Ảnh sinh bằng TextRecognitionDataGenerator với từ điển tiếng Việt Viet74K đã lọc (nhãn NFC, nhiều font có đủ dấu tiếng Việt). Chia **từ vựng** thành train/val/test/hard
rồi mới sinh ảnh, nên nhãn các tập không trùng nhau; tập hard dùng họ font chưa xuất hiện ở các tập khác cùng augmentation nặng (mờ, nhiễu, JPEG, nền phức tạp).
Bảng charset gồm ký tự tiếng Anh và tiếng Việt, tổng cộng 201 lớp đầu ra kể cả blank. Bộ dữ liệu dùng để huấn luyện `{run}` ({cfg['data']['root']}):

| Tập | Số mẫu | Nhãn khác nhau | Độ dài TB | Độ dài max |
|---|---:|---:|---:|---:|
{stats_rows}

**Huấn luyện `{run}`.** Từ đầu, H={m['img_height']}, batch {cfg['data']['batch_size']}, AdamW (lr tối đa {o['lr']}, weight decay {o['weight_decay']}), OneCycle, cắt gradient {o['clip_grad_norm']},
{cfg['train']['epochs']} epoch, seed {cfg['run']['seed']}, thiết bị `{summ['device']}`, trung bình {sec_ep:.0f} giây/epoch. Checkpoint tốt nhất theo val CER (epoch {summ['best_epoch']}, val CER {pc(summ['best_val_cer'])}%).
Lưu ý: tập hard dùng để đánh giá khả năng chống chịu, không dùng để chọn checkpoint.

**Các thực nghiệm phụ.** (a) Overfit 200 ảnh để kiểm tra mô hình học được; (b) chạy ablation tự động ở chế độ smoke (400 ảnh, vài epoch) chỉ để kiểm tra pipeline.

## 4. Kết quả

**Kết quả chính** (greedy và beam 10; thời gian là forward + decode trên 1 core CPU, batch 32, không tính đọc ảnh):

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh | ms decode |
|---|---|---:|---:|---:|---:|
{row_eval('vi_test', 'greedy', c_test['greedy'])}
{row_eval('vi_test', 'beam10', c_test['beam10'])}
{row_eval('vi_hard', 'greedy', c_hard['greedy'])}
{row_eval('vi_hard', 'beam10', c_hard['beam10'])}

Khoảng tin cậy 95% của greedy: test CER {pc(T['ci_cer'][0])}–{pc(T['ci_cer'][1])}%, Word Acc {pc(T['ci_word'][0])}–{pc(T['ci_word'][1])}%; hard CER {pc(an['vi_hard']['ci_cer'][0])}–{pc(an['vi_hard']['ci_cer'][1])}%.
Chênh lệch giữa beam 10 và greedy ({pc(float(c_test['greedy']['cer']) - float(c_test['beam10']['cer']))} điểm CER ở test) nằm trong các khoảng này nên **không đủ cơ sở kết luận beam search có ích**; trong khi đó beam 10 tốn thêm {float(c_test['beam10']['ms_decode']) - float(c_test['greedy']['ms_decode']):.2f} ms/ảnh để giải mã.

**Đường cong huấn luyện** (`outputs/figures/training_curves_mid_vi.png`): val CER thoát khỏi mức ~100% (cao nguyên đặc trưng của CTC) và xuống dưới 50% ở epoch {leave},
cuối cùng train loss {float(last['train_loss']):.3f} so với val loss {float(last['val_loss']):.3f}, cho thấy mô hình overfit nhẹ trên {st[0]['số mẫu']} ảnh huấn luyện.

**Kiểm tra overfit 200 ảnh** (CPU, cấu hình `small_cpu` ~{over['overfit200_en'][0]['params'] / 1e6:.1f} triệu tham số, lớn hơn `{run}`; train và val trên cùng 200 mẫu, nên chỉ chứng minh mô hình học được, không chứng minh khái quát hóa):

| Lần chạy | Epoch đầu có CER < 5% | CER cuối (%) | Word Acc cuối (%) | Số epoch |
|---|---:|---:|---:|---:|
{over_rows}

Trên 200 ảnh, fine-tune từ mô hình EN hội tụ nhanh hơn train từ đầu. Đây là bằng chứng về tốc độ tối ưu hóa; chưa có bằng chứng fine-tune cho CER tốt hơn trên dữ liệu chưa thấy.

**Độ nhạy với cách chia batch.** Đánh giá cùng checkpoint với batch 1 (không pad) thay vì batch 32 (có pad) cho CER test {pc(ev1[('vi_test', 'greedy')]['cer'])}% (so với {pc(c_test['greedy']['cer'])}%) và
CER hard {pc(ev1[('vi_hard', 'greedy')]['cer'])}% (so với {pc(c_hard['greedy']['cer'])}%), nhưng **{d_test}/{n_test} dự đoán ở test và {d_hard}/{n_hard} ở hard đổi kết quả**. Số liệu gộp ổn định nhưng từng mẫu
phụ thuộc vào phần pad (chủ yếu do BiLSTM chiều ngược đi qua vùng pad). `demo.py` xử lý từng ảnh riêng nên có thể khác `eval.py` ở một số ảnh.

**Ablation.** Chế độ smoke (`outputs/eval/ablation_smoke/`) chỉ xác nhận pipeline chạy từ huấn luyện, đánh giá đến bảng tổng hợp. Với 400 ảnh và vài epoch, mô hình chưa thoát cao nguyên CTC
nên các con số không cho phép kết luận gì; **không báo cáo kết quả ablation**.

## 5. Phân tích lỗi

Phân tích dựa trên căn chỉnh Levenshtein, mỗi lỗi thuộc đúng một loại (chi tiết và 10 ví dụ đúng/sai kèm ảnh trong `outputs/analysis/{run}/`).

| Nhóm lỗi | Test: số lỗi | Test: % | Hard: số lỗi | Hard: % |
|---|---:|---:|---:|---:|
{grp}

- **Dấu tiếng Việt** là nhóm lớn nhất ở test: {tone} lỗi dấu thanh và {modi} lỗi dấu phụ, trong đó mô hình gần như không nhầm mũ/móc/trăng với nhau mà thiếu hoặc thừa dấu phụ.
  Bỏ dấu làm CER giảm từ {pc(lad['đầy đủ']['cer'])}% xuống {pc(lad['bỏ dấu']['cer'])}% và Word Acc tăng từ {pc(lad['đầy đủ']['word_acc'], 1)}% lên {pc(lad['bỏ dấu']['word_acc'], 1)}%; đây chỉ là ước lượng phần lỗi do dấu.
- **"Thay thế khác"** ({pc(g(T, 'other') / E_t, 1)}% ở test) là lỗi nhận dạng rải rộng không thuộc nhóm nào, không cặp ký tự nào lặp lại thường xuyên. Nhóm "ký tự giống nhau" (O/0, l/1/I, S/5, rn/m) chỉ chiếm {pc(g(T, 'lookalike') / E_t, 1)}% ở test.
  Các cặp nhầm lẫn nhiều nhất: {top_pairs}.
- **Tương phản ảnh** có quan hệ mạnh với lỗi: CER {con('vi_test', 0)}% ở 25% ảnh tương phản thấp nhất so với {con('vi_test', 3)}% ở 25% cao nhất (test); {con('vi_hard', 0)}% so với {con('vi_hard', 3)}% (hard). Đây là tương quan, chưa phải nhân quả.
- **Tập hard** có tỉ lệ mất ký tự cao hơn nhiều ({pc(an['vi_hard']['D'] / an['vi_hard']['n_chars'], 1)}% số ký tự nhãn bị xóa, so với {pc(T['D'] / T['n_chars'], 1)}% ở test). Vì font lạ và nhiễu nặng đi cùng nhau trong tập này nên **không tách được** ảnh hưởng của từng yếu tố.
- Theo độ dài nhãn (test): {len_rows}. Word Accuracy giảm theo độ dài, điều được kỳ vọng khi lỗi ký tự cộng dồn trên chuỗi dài; chưa tách riêng ảnh hưởng của độ dài.

## 6. Hạn chế

- Chỉ dùng **dữ liệu sinh**; chưa đánh giá trên ảnh thật (chụp, nghiêng, nhiều dòng, nền phức tạp). Kết quả không suy ra được cho ảnh thật.
- Mô hình huấn luyện thật chỉ có {summ['params']:,} tham số, một lần chạy, một seed, một cấu hình (H=32), nên chưa có phương sai giữa các lần chạy; mô hình đầy đủ ({chk[32]['params']} tham số) mới chỉ được kiểm tra forward/backward và CTC hữu hạn (`outputs/checks/`), chưa từng được huấn luyện trên ảnh.
- **Ablation chưa chạy ở quy mô có ý nghĩa**; chưa so sánh được có/không BiLSTM, có/không augmentation, scratch/fine-tune trên dữ liệu đầy đủ.
- Tập val và test nhỏ ({ev[('vi_test', 'greedy')]['n']} mẫu test, {n_hard} mẫu hard); các chênh lệch dưới ~1 điểm CER (ví dụ greedy so với beam) không đáng tin.
- Dự đoán từng mẫu phụ thuộc vào phần pad của batch (xem mục 4).
- Các ngưỡng quy tắc đề xuất cải thiện và danh sách nhóm ký tự giống nhau trong phân tích lỗi là lựa chọn thủ công, chưa hiệu chỉnh.
- Nhánh mixed precision (AMP) và chạy GPU chưa được chạy lần nào; thời gian/ảnh chỉ đo trên CPU.

## 7. Hướng phát triển

Các nhận định dưới đây về những mô hình khác dựa trên mô tả công khai của từng dự án và **chưa được so sánh thực nghiệm trong dự án này**.

- **Mô hình ngôn ngữ / từ điển:** kết hợp beam search với mô hình ngôn ngữ hoặc từ điển tiếng Việt để sửa dấu. Phân tích lỗi cho thấy CER không dấu ({pc(lad['bỏ dấu']['cer'])}%) thấp hơn CER có dấu ({pc(lad['đầy đủ']['cer'])}%), nên có dư địa; không giúp được các chuỗi mã/số không có ngữ nghĩa.
- **Độ phân giải:** thử `img_height` 48 hoặc 64 (kiểm tra T ở H=48 cho thấy gần như hết mẫu sát giới hạn CTC) để nét dấu rõ hơn; so riêng số lỗi dấu thanh và dấu phụ trước/sau.
- **PARSeq:** dùng bộ giải mã Transformer huấn luyện với permutation language modeling, có thể tinh chỉnh kết quả theo ngữ cảnh; đáng so sánh trên cùng tập test để xem lợi ích với lỗi dấu.
- **TrOCR:** encoder ảnh + decoder văn bản Transformer tiền huấn luyện, có thể fine-tune cho tiếng Việt; cần nhiều tài nguyên hơn nhiều so với CRNN nhỏ.
- **VietOCR:** dự án OCR tiếng Việt có sẵn các mô hình VGG + Transformer/Seq2Seq và dữ liệu tiếng Việt; phù hợp làm mốc so sánh cho ảnh thật.
- **Dữ liệu và đánh giá:** bổ sung ảnh thật có gán nhãn; tách tập hard thành hai tập (font lạ riêng, nhiễu nặng riêng); chạy nhiều seed và bootstrap/kiểm định cho các so sánh; huấn luyện với `use_packing` hoặc đánh giá theo batch 1 để giảm phụ thuộc vào pad.

## 8. Tái lập và trạng thái

```bash
pip install -r requirements.txt
bash run_all_smoke.sh                    # toàn bộ pipeline với cấu hình nhỏ (CPU, vài phút)
python -m src.build_report --run {run}   # sinh lại báo cáo này từ outputs/
```

| Hạng mục | Trạng thái |
|---|---|
| Sinh dữ liệu smoke, bộ dữ liệu VI cỡ vừa ({st[0]['số mẫu']} train) | đã chạy |
| Sinh bộ dữ liệu đầy đủ (50k ảnh mỗi ngôn ngữ) | **chưa chạy** |
| Huấn luyện mô hình nhỏ `{run}` trên CPU | đã chạy |
| Huấn luyện đầy đủ trên GPU (AMP) cho pha EN, fine-tune VI, VI từ đầu | **chưa chạy** |
| Overfit 200 ảnh (EN, VI fine-tune, VI từ đầu) | đã chạy |
| Đánh giá greedy/beam, phân tích lỗi trên `{run}` | đã chạy |
| Ablation ở quy mô có ý nghĩa | **chưa chạy** (chỉ smoke) |
| Notebook Colab trên Colab thật | **chưa chạy** (chỉ kiểm tra mã bằng cách chạy các ô ở môi trường cục bộ) |
| Ảnh thật, so sánh PARSeq/TrOCR/VietOCR, mô hình ngôn ngữ, nhiều seed | **chưa chạy** |
"""
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mid_vi")
    ap.add_argument("--out", default=str(ROOT / "REPORT.md"))
    a = ap.parse_args()
    md = build(a.run, Path(a.out).parent)
    Path(a.out).write_text(md, encoding="utf-8")
    words = len(re.findall(r"\w+", md))
    print(f"Đã ghi {a.out}: {words} từ (ước lượng {words / 450:.1f} trang A4 gồm cả bảng)")


if __name__ == "__main__":
    main()
