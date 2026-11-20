### norn_en (ab_full_norn_en)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| en_test | greedy | 75.61 | 0.00 | 29.49 |
| en_test | beam5 | 70.31 | 0.00 | 50.02 |
| en_test | beam20 | 69.14 | 0.00 | 121.05 |
| en_hard | greedy | 95.33 | 0.00 | 23.12 |
| en_hard | beam5 | 94.04 | 0.00 | 41.58 |
| en_hard | beam20 | 93.54 | 0.00 | 106.06 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| en_test | greedy | 80 | 5.83 | 69.78 | 0.00 | 0.0 | 29.43 | 0.06 |
| en_test | beam5 | 80 | 8.70 | 61.61 | 0.00 | 0.0 | 29.43 | 20.59 |
| en_test | beam20 | 80 | 10.60 | 58.54 | 0.00 | 0.0 | 29.43 | 91.62 |
| en_hard | greedy | 80 | 6.56 | 88.77 | 0.00 | 0.0 | 23.07 | 0.05 |
| en_hard | beam5 | 80 | 14.12 | 78.93 | 0.99 | 0.0 | 23.07 | 18.50 |
| en_hard | beam20 | 80 | 16.20 | 76.44 | 0.89 | 0.0 | 23.07 | 82.99 |
