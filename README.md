# Nhận dạng chữ trong ảnh bằng CRNN + CTC (tiếng Anh và tiếng Việt có dấu)

CNN trích đặc trưng + BiLSTM mô hình hóa chuỗi + CTC loss để đọc một dòng chữ trong ảnh mà không cần căn chỉnh từng ký tự.
Dữ liệu sinh bằng TextRecognitionDataGenerator (TRDG); giải mã greedy và prefix beam search tự cài; đánh giá bằng CER và Word Accuracy; có phân tích lỗi (ký tự giống nhau, dấu tiếng Việt).
**Báo cáo:** [`REPORT.md`](REPORT.md) (mọi số liệu trong đó được sinh từ `outputs/` bằng `python -m src.build_report`).

## Trạng thái: cái gì đã chạy, cái gì chưa

| Hạng mục | Trạng thái |
|---|---|
| Pipeline cấu hình nhỏ trên CPU (`bash run_all_smoke.sh`), 130+ test | đã chạy |
| Mô hình nhỏ `mid_vi` (0,54 triệu tham số, 6000 ảnh VI, 40 epoch, CPU): CER test 7,55%, hard 50,49% | đã chạy |
| Overfit 200 ảnh (EN, VI fine-tune, VI từ đầu) | đã chạy |
| Greedy / beam search, phân tích lỗi trên `mid_vi` | đã chạy |
| Sinh bộ dữ liệu đầy đủ (50k ảnh/ngôn ngữ), huấn luyện đầy đủ trên GPU (AMP) | **chưa chạy** |
| Ablation ở quy mô có ý nghĩa (BiLSTM, augmentation, scratch/fine-tune) | **chưa chạy** (chỉ smoke để kiểm tra pipeline) |
| Notebook trên Colab thật | **chưa chạy** (các ô code đã được chạy cục bộ trong test) |
| Ảnh thật, so sánh PARSeq/TrOCR/VietOCR, mô hình ngôn ngữ, nhiều seed | **chưa chạy** |

Mô hình kèm theo (`checkpoints/mid_vi_best.pt`) chỉ biết ảnh sinh (chữ tối trên nền sáng, một dòng); trên ảnh thật có thể kém hơn nhiều.

## Cài đặt

Python 3.10+ (đã chạy với 3.12), PyTorch.
```bash
pip install -r requirements.txt
# Colab/Linux: thêm font để sinh dữ liệu tiếng Việt có dấu
apt-get install -y fonts-noto-core fonts-dejavu-core fonts-liberation fonts-freefont-ttf
```
`trdg==1.8.0` cần `setuptools<81` (đã ghi trong requirements) và được vá tương thích Pillow ≥ 10 trong `src/compat.py`.

## Dùng nhanh: nhận dạng một ảnh

```bash
python demo.py samples/vi_test_01.png                              # greedy, dùng checkpoints/mid_vi_best.pt
python demo.py --decoder beam10 --nbest 3 --verbose anh.png        # beam search, in xác suất và n-best
python demo.py thu_muc_anh/ --ckpt outputs/runs/vi_finetune/best.pt
```
Đầu ra: `<ảnh><TAB><chuỗi dự đoán>`. `--invert` cho chữ sáng trên nền tối. Notebook: `notebooks/colab_demo.ipynb`.
Lưu ý: dự đoán của từng ảnh có thể khác nhẹ so với `eval.py` chạy theo batch có pad (xem mục 4 của báo cáo); `demo.py` trùng với `eval.py --batch_size 1`.

## Pipeline từng bước

Chạy thử toàn bộ bằng cấu hình nhỏ (CPU, ~10 phút): `bash run_all_smoke.sh`. Chạy đầy đủ:

```bash
# 1. Dữ liệu (EN 50k/5k/5k/2k, VI tương tự; đổi số lượng trong configs/data.yaml)
bash run_stage1.sh                       # từ điển VI, kiểm tra font, sinh dữ liệu, charset, thống kê, ảnh mẫu, test
# 2. Kiểm tra mô hình và điều kiện T >= L+R của CTC
python -m src.check_model --height 48
# 3. Huấn luyện (GPU/Colab): pha (a) EN -> pha (b) fine-tune VI; đối chứng VI từ đầu
python -m src.train --config configs/full_gpu.yaml
python -m src.train --config configs/full_gpu_vi_finetune.yaml
python -m src.train --config configs/full_gpu_vi_scratch.yaml
python -m src.plot_curves en_base vi_finetune vi_scratch
# 4. Đánh giá: mỗi tập x (greedy, beam5/10/20), ghi outputs/eval/<run>/
python -m src.eval --ckpt vi_finetune
# 5. Phân tích lỗi: outputs/analysis/<run>/<tập>_<decoder>/report.md
python -m src.analyze_errors --tag vi_finetune --set vi_test --decoder greedy --val_cer <val CER của run>
# 6. Ablation tự động (cần counts.en.train_clean > 0 trong configs/data.yaml)
python -m src.ablation --mode full --alias base_en=en_base ft_vi=vi_finetune scratch_vi=vi_scratch --reuse
```
Mẹo: kiểm tra mô hình học được trước khi train dài: `python -m src.train --config configs/small_cpu.yaml --overfit 200 --set train.epochs=120 run.name=overfit200_en --overwrite`
(CTC có "cao nguyên": CER ~100% vài epoch đầu rồi mới giảm). Cột `infeasible_frac` trong `metrics.csv` phải ~0.

## Tái lập số liệu trong báo cáo (mô hình `mid_vi`)

```bash
bash run_stage1.sh --smoke                                   # tạo data/synth (smoke) và data/charset.txt
python data/generate_dataset.py --config configs/data_mid.yaml --lang vi     # data/synth_mid: 6000/300/800/400 ảnh VI
python -m src.train --config configs/small_cpu.yaml --set run.name=mid_vi data.root=data/synth_mid data.lang=vi \
    data.charset=data/charset.txt data.batch_size=32 "model.cnn.channels=[16,32,64,128]" model.rnn.hidden=64 optim.lr=0.004 \
    train.epochs=40 train.early_stop.patience=10 --overwrite                 # ~23 phút trên 1 core CPU
python -m src.eval --ckpt mid_vi --root data/synth_mid --sets vi/test vi/hard --decoders greedy beam10
python -m src.eval --ckpt mid_vi --root data/synth_mid --sets vi/test vi/hard --decoders greedy --batch_size 1 --tag mid_vi_bs1
python -m src.analyze_errors --tag mid_vi --set vi_test --decoder greedy --root data/synth_mid
python -m src.analyze_errors --tag mid_vi --set vi_hard --decoder greedy --root data/synth_mid
python data/stats.py --root data/synth_mid --lang vi --out outputs/stats/mid
python -m src.check_model --height 32 > outputs/checks/model_h32.txt     # tương tự --height 48
python -m src.build_report --run mid_vi
```
Seed cố định nhưng **chưa kiểm tra tái lập từng bit**; kết quả có thể lệch nhẹ giữa các máy/phiên bản PyTorch. `charset.txt` phụ thuộc dữ liệu smoke đã sinh trước đó.
`build_report` đọc các file có tên cố định trong `outputs/` (xem docstring), nên với run khác cần tạo đủ các file tương ứng.

## Cấu trúc thư mục

```
configs/        data.yaml, data_mid.yaml, model*.yaml, small_cpu.yaml, full_gpu*.yaml
data/           make_vi_wordlist.py, check_fonts.py, generate_dataset.py, build_charset.py, stats.py, show_samples.py, vi_words.txt
src/            dataset.py, model.py, ctc_decode.py, ctc_check.py, metrics.py, train.py, eval.py, ablation.py,
                analyze_errors.py, plot_curves.py, check_model.py, build_report.py, text_utils.py, compat.py
tests/          test_data.py, test_model.py, test_train.py, test_decode_eval.py, test_analyze.py, test_demo.py
checkpoints/    mid_vi_best.pt (mô hình nhỏ dùng cho demo)
samples/        12 ảnh mẫu + labels.tsv
notebooks/      colab_demo.ipynb
outputs/        runs/ (log, metrics.csv), eval/, analysis/, figures/, stats/, checks/   (checkpoint .pt không kèm trong gói)
demo.py  run_stage1.sh  run_all_smoke.sh  REPORT.md  requirements.txt
```

## Test

```bash
python -m pytest -q tests          # cần đã chạy bước dữ liệu (run_all_smoke.sh); test thiếu dữ liệu sẽ tự bỏ qua
```
Gồm: CER/S-D-I (đối chiếu jiwer, rapidfuzz), prefix beam search (đối chiếu liệt kê toàn bộ alignment và pyctcdecode), NFC/charset, mô hình/CTC, huấn luyện (overfit, resume, mở rộng lớp Linear), phân loại lỗi dấu, demo và báo cáo.

## Lưu ý về dữ liệu

Từ điển tiếng Việt lấy từ `duyet/vietnamese-wordlist` (Viet74K) và font từ TRDG/hệ thống: **kiểm tra giấy phép** trước khi phát hành lại dữ liệu hoặc mô hình.
`dicts/vi.txt` đi kèm TRDG không phải tiếng Việt có dấu nên không được dùng.
