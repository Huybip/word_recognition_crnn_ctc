#!/usr/bin/env bash
# Giai đoạn 1 - chạy tuần tự. Dùng: bash run_stage1.sh [--smoke]
set -e
SMOKE="$1"                       # --smoke: bộ nhỏ để kiểm tra pipeline
python data/make_vi_wordlist.py  # tải + làm sạch từ điển VI (cần internet)
python data/check_fonts.py --sheet
python data/generate_dataset.py --lang both --overwrite $SMOKE
python data/build_charset.py
python data/stats.py
python data/show_samples.py --splits train hard
pytest -q tests/test_data.py
