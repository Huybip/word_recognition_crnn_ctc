# Nhận dạng chữ trong ảnh bằng CRNN + CTC (tiếng Anh và tiếng Việt có dấu)

> Mọi số liệu trong tài liệu này được `src/build_report.py` đọc từ thư mục `outputs/`. Toàn bộ thực nghiệm chạy trên **dữ liệu sinh (TRDG)**,
> trên **CPU 1 core**, với **một mô hình nhỏ** và **một lần chạy (một seed)**. Chưa có thực nghiệm trên ảnh thật hay trên GPU.

## Tóm tắt

Dự án cài đặt và đánh giá mô hình CRNN (CNN + BiLSTM) huấn luyện bằng CTC để đọc một dòng chữ trong ảnh, có phần thử riêng cho tiếng Việt có dấu.
Mô hình nhỏ `mid_vi` (539,705 tham số, 40 epoch, 23 phút trên CPU) đạt **CER 7.55%** và
**Word Accuracy 63.38%** trên tập test tiếng Việt (800 mẫu, cùng phân phối với tập huấn luyện), nhưng chỉ đạt CER 50.49%
trên tập hard (font chưa từng thấy và nhiễu/mờ nặng). Lỗi dấu chiếm 44.2% số lỗi trên tập test; bỏ dấu khỏi nhãn và dự đoán làm CER giảm từ
7.55% xuống 4.18%. Beam search (beam 10) chỉ cải thiện CER 0.07 điểm trên tập test,
nằm trong khoảng nhiễu. Các ablation (BiLSTM, augmentation, fine-tune EN→VI) chỉ được kiểm tra ở mức pipeline chạy thông, **chưa có kết quả có ý nghĩa**.

## 1. Giới thiệu

Bài toán là nhận dạng dòng chữ trong ảnh (line-level text recognition). Cách làm kinh điển cho một dòng chữ có độ dài thay đổi là CRNN + CTC:
CNN trích đặc trưng, RNN hai chiều mô hình hóa chuỗi theo chiều ngang, và hàm mất mát CTC cho phép huấn luyện mà **không cần căn chỉnh từng ký tự** với ảnh.
Mục tiêu của dự án: (1) chuẩn bị dữ liệu ảnh chữ, (2) cài đặt CNN + BiLSTM + CTC, (3) giải mã bằng greedy và beam search, (4) đánh giá bằng CER và Word Accuracy,
(5) thử trên chữ tiếng Việt có dấu và (6) phân tích lỗi (ký tự giống nhau, dấu).

## 2. Phương pháp

**Kiến trúc.** CNN kiểu VGG nhỏ gồm 6 lớp Conv3×3 + BatchNorm + ReLU xen MaxPool, sau đó một Conv có kernel cao H/16 đưa chiều cao về 1, giữ chiều rộng làm
trục thời gian với T = W/4. Tiếp theo là 2 lớp BiLSTM (hidden 256), lớp Linear sang số lớp = số ký tự + 1 (blank ở index 0) và log-softmax. Cấu hình đầy đủ có
8,283,785 tham số; mô hình đã huấn luyện (`mid_vi`) dùng kênh [16, 32, 64, 128] và hidden 64 để chạy được trên CPU.

**CTC và điều kiện về độ dài.** CTC chỉ căn chỉnh được khi T ≥ L + R (L là độ dài nhãn, R là số cặp ký tự giống nhau liền kề, vì giữa hai ký tự lặp cần một blank).
Kiểm tra trên 1280 mẫu (bộ smoke, nhãn dài nhất 25 ký tự): ở H=32, stride 4 có 0 mẫu vi phạm điều kiện chính xác nhưng 99 mẫu dưới ngưỡng bảo thủ T ≥ 2L+1
(tập trung ở font hẹp); ở H=48, stride 4 chỉ còn 1 mẫu. `zero_infinity=True` sẽ âm thầm bỏ qua mẫu vô nghiệm nên mã huấn luyện đếm và ghi lại tỉ lệ này.

**Giải mã.** Greedy (argmax từng bước, gộp ký tự lặp, bỏ blank) và prefix beam search tự cài, tính trong không gian log, cộng xác suất mọi alignment của cùng một chuỗi.
Cài đặt được đối chiếu với liệt kê toàn bộ alignment (brute-force) trên các ca nhỏ: beam đủ lớn cho đúng chuỗi và đúng xác suất. Beam=1 **không** luôn trùng greedy.

**Chỉ số.** CER = (S + D + I) / N gộp toàn tập (N là tổng số ký tự nhãn), Word Accuracy là tỉ lệ dòng khớp hoàn toàn nhãn. Khoảng tin cậy 95% dùng bootstrap theo mẫu (1000 lần).

## 3. Thiết lập thí nghiệm

**Dữ liệu.** Ảnh sinh bằng TextRecognitionDataGenerator với từ điển tiếng Việt Viet74K đã lọc (nhãn NFC, nhiều font có đủ dấu tiếng Việt). Chia **từ vựng** thành train/val/test/hard
rồi mới sinh ảnh, nên nhãn các tập không trùng nhau; tập hard dùng họ font chưa xuất hiện ở các tập khác cùng augmentation nặng (mờ, nhiễu, JPEG, nền phức tạp).
Bảng charset gồm ký tự tiếng Anh và tiếng Việt, tổng cộng 201 lớp đầu ra kể cả blank. Bộ dữ liệu dùng để huấn luyện `mid_vi` (data/synth_mid):

| Tập | Số mẫu | Nhãn khác nhau | Độ dài TB | Độ dài max |
|---|---:|---:|---:|---:|
| train | 6000 | 5841 | 10.1 | 25 |
| val | 300 | 300 | 10.2 | 21 |
| test | 800 | 800 | 10.5 | 25 |
| hard | 400 | 400 | 10.4 | 23 |

**Huấn luyện `mid_vi`.** Từ đầu, H=32, batch 32, AdamW (lr tối đa 0.004, weight decay 0.01), OneCycle, cắt gradient 5.0,
40 epoch, seed 42, thiết bị `cpu`, trung bình 35 giây/epoch. Checkpoint tốt nhất theo val CER (epoch 39, val CER 7.41%).
Lưu ý: tập hard dùng để đánh giá khả năng chống chịu, không dùng để chọn checkpoint.

**Các thực nghiệm phụ.** (a) Overfit 200 ảnh để kiểm tra mô hình học được; (b) chạy ablation tự động ở chế độ smoke (400 ảnh, vài epoch) chỉ để kiểm tra pipeline.

## 4. Kết quả

**Kết quả chính** (greedy và beam 10; thời gian là forward + decode trên 1 core CPU, batch 32, không tính đọc ảnh):

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh | ms decode |
|---|---|---:|---:|---:|---:|
| vi_test | greedy | 7.55 | 63.38 | 1.56 | 0.08 |
| vi_test | beam10 | 7.47 | 63.50 | 3.03 | 1.56 |
| vi_hard | greedy | 50.49 | 13.25 | 1.27 | 0.02 |
| vi_hard | beam10 | 49.94 | 13.50 | 8.38 | 7.14 |

Khoảng tin cậy 95% của greedy: test CER 6.47–8.54%, Word Acc 60.25–66.75%; hard CER 47.02–54.10%.
Chênh lệch giữa beam 10 và greedy (0.07 điểm CER ở test) nằm trong các khoảng này nên **không đủ cơ sở kết luận beam search có ích**; trong khi đó beam 10 tốn thêm 1.47 ms/ảnh để giải mã.

**Đường cong huấn luyện** (`outputs/figures/training_curves_mid_vi.png`): val CER thoát khỏi mức ~100% (cao nguyên đặc trưng của CTC) và xuống dưới 50% ở epoch 6,
cuối cùng train loss 0.012 so với val loss 0.360, cho thấy mô hình overfit nhẹ trên 6000 ảnh huấn luyện.

**Kiểm tra overfit 200 ảnh** (CPU, cấu hình `small_cpu` ~2.1 triệu tham số, lớn hơn `mid_vi`; train và val trên cùng 200 mẫu, nên chỉ chứng minh mô hình học được, không chứng minh khái quát hóa):

| Lần chạy | Epoch đầu có CER < 5% | CER cuối (%) | Word Acc cuối (%) | Số epoch |
|---|---:|---:|---:|---:|
| overfit200_en | 66 | 0.43 | 97.0 | 120 |
| overfit200_vi_ft | 21 | 0.29 | 97.5 | 100 |
| overfit200_vi_scratch | 84 | 4.77 | 72.0 | 100 |

Trên 200 ảnh, fine-tune từ mô hình EN hội tụ nhanh hơn train từ đầu. Đây là bằng chứng về tốc độ tối ưu hóa; chưa có bằng chứng fine-tune cho CER tốt hơn trên dữ liệu chưa thấy.

**Độ nhạy với cách chia batch.** Đánh giá cùng checkpoint với batch 1 (không pad) thay vì batch 32 (có pad) cho CER test 7.49% (so với 7.55%) và
CER hard 50.59% (so với 50.49%), nhưng **49/800 dự đoán ở test và 165/400 ở hard đổi kết quả**. Số liệu gộp ổn định nhưng từng mẫu
phụ thuộc vào phần pad (chủ yếu do BiLSTM chiều ngược đi qua vùng pad). `demo.py` xử lý từng ảnh riêng nên có thể khác `eval.py` ở một số ảnh.

**Ablation.** Chế độ smoke (`outputs/eval/ablation_smoke/`) chỉ xác nhận pipeline chạy từ huấn luyện, đánh giá đến bảng tổng hợp. Với 400 ảnh và vài epoch, mô hình chưa thoát cao nguyên CTC
nên các con số không cho phép kết luận gì; **không báo cáo kết quả ablation**.

## 5. Phân tích lỗi

Phân tích dựa trên căn chỉnh Levenshtein, mỗi lỗi thuộc đúng một loại (chi tiết và 10 ví dụ đúng/sai kèm ảnh trong `outputs/analysis/mid_vi/`).

| Nhóm lỗi | Test: số lỗi | Test: % | Hard: số lỗi | Hard: % |
|---|---:|---:|---:|---:|
| Lỗi dấu tiếng Việt | 280 | 44.2 | 242 | 11.5 |
| Thay thế khác | 201 | 31.8 | 978 | 46.5 |
| Mất ký tự | 83 | 13.1 | 612 | 29.1 |
| Hoa/thường | 26 | 4.1 | 31 | 1.5 |
| Khoảng trắng | 22 | 3.5 | 171 | 8.1 |
| Thừa ký tự | 11 | 1.7 | 49 | 2.3 |
| Ký tự giống nhau | 10 | 1.6 | 21 | 1.0 |

- **Dấu tiếng Việt** là nhóm lớn nhất ở test: 178 lỗi dấu thanh và 73 lỗi dấu phụ, trong đó mô hình gần như không nhầm mũ/móc/trăng với nhau mà thiếu hoặc thừa dấu phụ.
  Bỏ dấu làm CER giảm từ 7.55% xuống 4.18% và Word Acc tăng từ 63.4% lên 82.8%; đây chỉ là ước lượng phần lỗi do dấu.
- **"Thay thế khác"** (31.8% ở test) là lỗi nhận dạng rải rộng không thuộc nhóm nào, không cặp ký tự nào lặp lại thường xuyên. Nhóm "ký tự giống nhau" (O/0, l/1/I, S/5, rn/m) chỉ chiếm 1.6% ở test.
  Các cặp nhầm lẫn nhiều nhất: ` `→`∅` (10); `n`→`∅` (7); `ỉ`→`i` (4); `t`→`∅` (4); `ỗ`→`ổ` (4).
- **Tương phản ảnh** có quan hệ mạnh với lỗi: CER 13.3% ở 25% ảnh tương phản thấp nhất so với 3.5% ở 25% cao nhất (test); 70.8% so với 25.6% (hard). Đây là tương quan, chưa phải nhân quả.
- **Tập hard** có tỉ lệ mất ký tự cao hơn nhiều (16.9% số ký tự nhãn bị xóa, so với 1.1% ở test). Vì font lạ và nhiễu nặng đi cùng nhau trong tập này nên **không tách được** ảnh hưởng của từng yếu tố.
- Theo độ dài nhãn (test): 1-5 ký tự: 14.0%; 6-10 ký tự: 6.2%; 11-15 ký tự: 5.9%; 16-20 ký tự: 9.3%; 21+ ký tự: 9.9%. Word Accuracy giảm theo độ dài, điều được kỳ vọng khi lỗi ký tự cộng dồn trên chuỗi dài; chưa tách riêng ảnh hưởng của độ dài.

## 6. Hạn chế

- Chỉ dùng **dữ liệu sinh**; chưa đánh giá trên ảnh thật (chụp, nghiêng, nhiều dòng, nền phức tạp). Kết quả không suy ra được cho ảnh thật.
- Mô hình huấn luyện thật chỉ có 539,705 tham số, một lần chạy, một seed, một cấu hình (H=32), nên chưa có phương sai giữa các lần chạy; mô hình đầy đủ (8,283,785 tham số) mới chỉ được kiểm tra forward/backward và CTC hữu hạn (`outputs/checks/`), chưa từng được huấn luyện trên ảnh.
- **Ablation chưa chạy ở quy mô có ý nghĩa**; chưa so sánh được có/không BiLSTM, có/không augmentation, scratch/fine-tune trên dữ liệu đầy đủ.
- Tập val và test nhỏ (800 mẫu test, 400 mẫu hard); các chênh lệch dưới ~1 điểm CER (ví dụ greedy so với beam) không đáng tin.
- Dự đoán từng mẫu phụ thuộc vào phần pad của batch (xem mục 4).
- Các ngưỡng quy tắc đề xuất cải thiện và danh sách nhóm ký tự giống nhau trong phân tích lỗi là lựa chọn thủ công, chưa hiệu chỉnh.
- Nhánh mixed precision (AMP) và chạy GPU chưa được chạy lần nào; thời gian/ảnh chỉ đo trên CPU.

## 7. Hướng phát triển

Các nhận định dưới đây về những mô hình khác dựa trên mô tả công khai của từng dự án và **chưa được so sánh thực nghiệm trong dự án này**.

- **Mô hình ngôn ngữ / từ điển:** kết hợp beam search với mô hình ngôn ngữ hoặc từ điển tiếng Việt để sửa dấu. Phân tích lỗi cho thấy CER không dấu (4.18%) thấp hơn CER có dấu (7.55%), nên có dư địa; không giúp được các chuỗi mã/số không có ngữ nghĩa.
- **Độ phân giải:** thử `img_height` 48 hoặc 64 (kiểm tra T ở H=48 cho thấy gần như hết mẫu sát giới hạn CTC) để nét dấu rõ hơn; so riêng số lỗi dấu thanh và dấu phụ trước/sau.
- **PARSeq:** dùng bộ giải mã Transformer huấn luyện với permutation language modeling, có thể tinh chỉnh kết quả theo ngữ cảnh; đáng so sánh trên cùng tập test để xem lợi ích với lỗi dấu.
- **TrOCR:** encoder ảnh + decoder văn bản Transformer tiền huấn luyện, có thể fine-tune cho tiếng Việt; cần nhiều tài nguyên hơn nhiều so với CRNN nhỏ.
- **VietOCR:** dự án OCR tiếng Việt có sẵn các mô hình VGG + Transformer/Seq2Seq và dữ liệu tiếng Việt; phù hợp làm mốc so sánh cho ảnh thật.
- **Dữ liệu và đánh giá:** bổ sung ảnh thật có gán nhãn; tách tập hard thành hai tập (font lạ riêng, nhiễu nặng riêng); chạy nhiều seed và bootstrap/kiểm định cho các so sánh; huấn luyện với `use_packing` hoặc đánh giá theo batch 1 để giảm phụ thuộc vào pad.

## 8. Tái lập và trạng thái

```bash
pip install -r requirements.txt
bash run_all_smoke.sh                    # toàn bộ pipeline với cấu hình nhỏ (CPU, vài phút)
python -m src.build_report --run mid_vi   # sinh lại báo cáo này từ outputs/
```

| Hạng mục | Trạng thái |
|---|---|
| Sinh dữ liệu smoke, bộ dữ liệu VI cỡ vừa (6000 train) | đã chạy |
| Sinh bộ dữ liệu đầy đủ (50k ảnh mỗi ngôn ngữ) | **chưa chạy** |
| Huấn luyện mô hình nhỏ `mid_vi` trên CPU | đã chạy |
| Huấn luyện đầy đủ trên GPU (AMP) cho pha EN, fine-tune VI, VI từ đầu | **chưa chạy** |
| Overfit 200 ảnh (EN, VI fine-tune, VI từ đầu) | đã chạy |
| Đánh giá greedy/beam, phân tích lỗi trên `mid_vi` | đã chạy |
| Ablation ở quy mô có ý nghĩa | **chưa chạy** (chỉ smoke) |
| Notebook Colab trên Colab thật | **chưa chạy** (chỉ kiểm tra mã bằng cách chạy các ô ở môi trường cục bộ) |
| Ảnh thật, so sánh PARSeq/TrOCR/VietOCR, mô hình ngôn ngữ, nhiều seed | **chưa chạy** |
