# Biên bản rà soát cuối (Giai đoạn 6)

## Cách kiểm tra
Giải nén gói zip vào một thư mục **sạch** (không có dữ liệu sinh, không có checkpoint huấn luyện, chỉ có `checkpoints/mid_vi_best.pt`)
rồi chạy `bash run_all_smoke.sh` với cấu hình nhỏ. Môi trường: Python 3.12.3, PyTorch 2.15.0, CPU 1 core, không GPU.

## Kết quả
- Mọi bước chạy hết, không lỗi: từ điển + font + sinh dữ liệu smoke, charset, thống kê, kiểm tra mô hình (CTC hữu hạn), huấn luyện EN rồi fine-tune VI (mỗi pha 2 epoch),
  đánh giá greedy/beam, phân tích lỗi, ablation chế độ smoke (6 thí nghiệm), demo, test.
- **133 test đạt, 0 test bị bỏ qua** trong thư mục sạch.
- Charset sinh lại từ dữ liệu smoke **khớp** với charset lưu trong `mid_vi_best.pt` (200 ký tự).
- `REPORT.md` sinh lại từ `outputs/` bằng `python -m src.build_report` **giống hệt từng byte** bản gốc.
- `demo.py` trên ảnh mẫu cho cùng kết quả với `eval.py --batch_size 1` (có test).
- Cảnh báo duy nhất trong log là lời nhắc có chủ đích rằng số liệu chế độ smoke không có ý nghĩa.

Số liệu của các lần chạy smoke (ví dụ CER 100% sau 2 epoch trên 400 ảnh) chỉ chứng minh pipeline thông, **không** nói gì về độ chính xác.

## Chưa chạy (không được coi là đã kiểm chứng)
1. Sinh bộ dữ liệu đầy đủ (50k ảnh mỗi ngôn ngữ) và huấn luyện đầy đủ trên GPU, kể cả nhánh mixed precision (AMP). Không có GPU ở môi trường phát triển.
2. Ablation ở quy mô có ý nghĩa: có/không BiLSTM, có/không augmentation, greedy vs beam, scratch vs fine-tune. Mới chạy chế độ smoke.
3. Mô hình đầy đủ 8,3 triệu tham số trên ảnh thật: mới kiểm tra forward/backward và CTC, chưa huấn luyện.
4. Notebook trên Colab thật. Mới chạy các ô code ở môi trường cục bộ (có test), không chạy giao diện Colab (`files.upload`).
5. Ảnh thật (chụp, nghiêng, nhiều dòng), so sánh với PARSeq/TrOCR/VietOCR, mô hình ngôn ngữ, nhiều seed, kiểm định ý nghĩa thống kê.
6. Tái lập từng bit của huấn luyện (seed cố định nhưng chưa kiểm tra giữa các máy/phiên bản PyTorch).

## Giới hạn đã biết của kết quả đang có
- Mô hình `mid_vi` chỉ gồm 0,54 triệu tham số, huấn luyện 6000 ảnh sinh, một seed. CER test 7,55% là trên dữ liệu cùng phân phối; tập hard (font lạ + nhiễu nặng) CER 50,49%.
- Dự đoán từng ảnh phụ thuộc vào phần pad của batch: 49/800 ảnh test đổi kết quả giữa batch 32 và batch 1 (CER gộp gần như không đổi: 7,55% so với 7,49%).
- Chênh lệch greedy/beam (0,08 điểm CER) nằm trong khoảng tin cậy, không kết luận được beam search có ích.
