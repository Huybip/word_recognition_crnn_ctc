### Kết quả: overfit200_vi_ft

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| vi_test | greedy | 51.38 | 1.25 | 3.68 |
| vi_test | beam10 | 51.38 | 1.25 | 25.28 |
| vi_hard | greedy | 78.25 | 0.00 | 3.72 |
| vi_hard | beam10 | 78.01 | 0.00 | 39.39 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vi_test | greedy | 80 | 34.17 | 14.57 | 2.64 | 0.0 | 3.63 | 0.06 |
| vi_test | beam10 | 80 | 33.67 | 14.95 | 2.76 | 0.0 | 3.63 | 21.66 |
| vi_hard | greedy | 80 | 42.67 | 34.40 | 1.18 | 0.0 | 3.66 | 0.06 |
| vi_hard | beam10 | 80 | 43.74 | 32.98 | 1.30 | 0.0 | 3.66 | 35.74 |
