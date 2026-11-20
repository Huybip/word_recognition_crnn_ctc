#!/usr/bin/env bash
# Chạy TOÀN BỘ pipeline với cấu hình nhỏ (CPU, khoảng 8-10 phút) để kiểm tra mọi bước hoạt động.
# Số liệu sinh ra ở đây (bộ smoke 400 ảnh, vài epoch) chỉ để kiểm tra pipeline, KHÔNG có ý nghĩa về độ chính xác.
# Dùng: bash run_all_smoke.sh
set -euo pipefail
step() { echo; echo "=============== $* ==============="; }
export PYTHONWARNINGS=ignore

step "1. Dữ liệu"
python data/make_vi_wordlist.py || echo "(không tải được từ điển, dùng data/vi_words.txt có sẵn)"
python data/check_fonts.py --sheet
python data/generate_dataset.py --lang both --smoke --overwrite
python data/build_charset.py
python data/stats.py --out outputs/stats/smoke
step "2. Mô hình"
python -m src.check_model
step "3. Huấn luyện (EN rồi fine-tune VI, mỗi pha 2 epoch)"
python -m src.train --config configs/small_cpu.yaml --set run.name=smoke_en --overwrite
python -m src.train --config configs/small_cpu.yaml --set run.name=smoke_vi_ft data.lang=vi data.charset=data/charset.txt \
    init_from=outputs/runs/smoke_en/best.pt --overwrite
step "4. Đánh giá (greedy, beam)"
python -m src.eval --ckpt smoke_vi_ft --decoders greedy beam5
step "5. Phân tích lỗi"
python -m src.analyze_errors --tag smoke_vi_ft --set vi_test --decoder greedy
step "6. Ablation (chế độ smoke, 3 epoch mỗi thí nghiệm)"
python -m src.ablation --mode smoke --epochs 3 --decoders greedy beam5
step "7. Demo"
python demo.py --verbose --decoder beam5 samples/vi_test_00.png samples/vi_test_01.png
step "8. Test"
python -m pytest -q tests
echo; echo "HOÀN TẤT: pipeline chạy thông."
