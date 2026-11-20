### noaug_en (ab_full_noaug_en)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| en_test | greedy | 100.00 | 0.00 | 30.81 |
| en_test | beam5 | 96.08 | 0.00 | 51.78 |
| en_test | beam20 | 96.08 | 0.00 | 128.43 |
| en_hard | greedy | 100.00 | 0.00 | 23.83 |
| en_hard | beam5 | 96.52 | 0.00 | 41.54 |
| en_hard | beam20 | 96.52 | 0.00 | 106.24 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| en_test | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 30.75 | 0.06 |
| en_test | beam5 | 80 | 2.12 | 93.96 | 0.00 | 0.0 | 30.75 | 21.03 |
| en_test | beam20 | 80 | 2.12 | 93.96 | 0.00 | 0.0 | 30.75 | 97.68 |
| en_hard | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 23.77 | 0.07 |
| en_hard | beam5 | 80 | 2.98 | 93.54 | 0.00 | 0.0 | 23.77 | 17.77 |
| en_hard | beam20 | 80 | 2.98 | 93.54 | 0.00 | 0.0 | 23.77 | 82.47 |
