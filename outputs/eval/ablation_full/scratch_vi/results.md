### scratch_vi (vi_scratch)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| vi_test | greedy | 100.00 | 0.00 | 23.81 |
| vi_test | beam5 | 97.61 | 0.00 | 74.80 |
| vi_test | beam20 | 90.45 | 0.00 | 388.18 |
| vi_hard | greedy | 100.00 | 0.00 | 21.67 |
| vi_hard | beam5 | 94.44 | 0.00 | 69.10 |
| vi_hard | beam20 | 89.83 | 0.00 | 345.55 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vi_test | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 23.74 | 0.07 |
| vi_test | beam5 | 80 | 0.00 | 97.61 | 0.00 | 0.0 | 23.74 | 51.06 |
| vi_test | beam20 | 80 | 0.75 | 89.70 | 0.00 | 0.0 | 23.74 | 364.44 |
| vi_hard | greedy | 80 | 0.00 | 100.00 | 0.00 | 0.0 | 21.61 | 0.06 |
| vi_hard | beam5 | 80 | 0.24 | 94.21 | 0.00 | 0.0 | 21.61 | 47.49 |
| vi_hard | beam20 | 80 | 0.47 | 89.36 | 0.00 | 0.0 | 21.61 | 323.94 |
