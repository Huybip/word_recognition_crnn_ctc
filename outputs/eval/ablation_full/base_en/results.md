### base_en (en_base)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| en_test | greedy | 99.05 | 0.00 | 30.77 |
| en_test | beam5 | 170.73 | 0.00 | 51.68 |
| en_test | beam20 | 170.73 | 0.00 | 122.68 |
| en_hard | greedy | 99.40 | 0.00 | 25.68 |
| en_hard | beam5 | 138.77 | 0.00 | 44.41 |
| en_hard | beam20 | 138.77 | 0.00 | 106.56 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| en_test | greedy | 80 | 9.65 | 89.40 | 0.00 | 0.0 | 30.72 | 0.06 |
| en_test | beam5 | 80 | 97.67 | 0.42 | 72.64 | 0.0 | 30.72 | 20.96 |
| en_test | beam20 | 80 | 97.67 | 0.42 | 72.64 | 0.0 | 30.72 | 91.96 |
| en_hard | greedy | 80 | 8.65 | 90.76 | 0.00 | 0.0 | 25.63 | 0.05 |
| en_hard | beam5 | 80 | 97.42 | 1.39 | 39.96 | 0.0 | 25.63 | 18.78 |
| en_hard | beam20 | 80 | 97.42 | 1.39 | 39.96 | 0.0 | 25.63 | 80.93 |
