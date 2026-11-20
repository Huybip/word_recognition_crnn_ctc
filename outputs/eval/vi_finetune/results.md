### Kết quả: vi_finetune

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| vi_test | greedy | 95.60 | 0.00 | 23.42 |
| vi_test | beam10 | 160.18 | 0.00 | 139.78 |
| vi_hard | greedy | 94.92 | 0.00 | 26.10 |
| vi_hard | beam10 | 134.75 | 0.00 | 131.83 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vi_test | greedy | 80 | 16.46 | 79.15 | 0.00 | 0.0 | 23.35 | 0.07 |
| vi_test | beam10 | 80 | 93.72 | 0.00 | 66.46 | 0.0 | 23.35 | 116.43 |
| vi_hard | greedy | 80 | 17.02 | 77.90 | 0.00 | 0.0 | 26.03 | 0.07 |
| vi_hard | beam10 | 80 | 93.03 | 0.24 | 41.49 | 0.0 | 26.03 | 105.80 |
