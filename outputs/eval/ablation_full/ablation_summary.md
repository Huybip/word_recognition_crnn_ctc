# Ablation (full)

Cấu hình: full_gpu.yaml, epochs=theo config, decoders=['greedy', 'beam5', 'beam20']

## Ablation 1: có / không BiLSTM (EN, greedy)

| Biến thể | Tham số | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) |
|---|---:|---:|---:|---:|---:|
| CNN + BiLSTM + Linear | 8,477,187 | 99.05 | 0.00 | 99.40 | 0.00 |
| CNN + Linear (không RNN) | 5,323,267 | 75.61 | 0.00 | 95.33 | 0.00 |

## Ablation 2: có / không augmentation (EN, greedy, cùng số mẫu huấn luyện)

| Biến thể | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) |
|---|---:|---:|---:|---:|
| có augmentation (train) | 100.00 | 0.00 | 100.00 | 0.00 |
| không augmentation (train_clean) | 100.00 | 0.00 | 100.00 | 0.00 |

## Ablation 3: greedy vs beam search (mô hình base_en)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh (forward+decode) | ms decode |
|---|---|---:|---:|---:|---:|
| en_test | greedy | 99.05 | 0.00 | 30.77 | 0.06 |
| en_test | beam5 | 170.73 | 0.00 | 51.68 | 20.96 |
| en_test | beam20 | 170.73 | 0.00 | 122.68 | 91.96 |
| en_hard | greedy | 99.40 | 0.00 | 25.68 | 0.05 |
| en_hard | beam5 | 138.77 | 0.00 | 44.41 | 18.78 |
| en_hard | beam20 | 138.77 | 0.00 | 106.56 | 80.93 |

## Ablation 4: fine-tune từ EN vs train VI từ đầu (greedy)

| Biến thể | CER test (%) | Word Acc test (%) | CER hard (%) | Word Acc hard (%) | epoch tốt nhất | val CER tốt nhất (%) |
|---|---:|---:|---:|---:|---:|---:|
| fine-tune từ EN | 95.60 | 0.00 | 94.92 | 0.00 | 2 | 94.33 |
| từ đầu | 100.00 | 0.00 | 100.00 | 0.00 | 30 | 100.00 |
