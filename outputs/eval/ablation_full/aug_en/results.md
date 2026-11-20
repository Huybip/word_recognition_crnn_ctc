### aug_en (ab_full_aug_en)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| en_test | greedy | 100.00 | 0.00 | 29.32 |
| en_test | beam5 | 96.29 | 0.00 | 50.39 |
| en_test | beam20 | 96.29 | 0.00 | 127.28 |
| en_hard | greedy | 100.00 | 0.00 | 26.40 |
| en_hard | beam5 | 97.22 | 0.00 | 44.21 |
| en_hard | beam20 | 97.22 | 0.00 | 110.21 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| en_test | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 29.25 | 0.07 |
| en_test | beam5 | 80 | 2.23 | 94.06 | 0.00 | 0.0 | 29.25 | 21.14 |
| en_test | beam20 | 80 | 2.23 | 94.06 | 0.00 | 0.0 | 29.25 | 98.03 |
| en_hard | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 26.34 | 0.06 |
| en_hard | beam5 | 80 | 2.39 | 94.83 | 0.00 | 0.0 | 26.34 | 17.86 |
| en_hard | beam20 | 80 | 2.39 | 94.83 | 0.00 | 0.0 | 26.34 | 83.87 |
