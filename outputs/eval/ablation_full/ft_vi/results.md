### ft_vi (vi_finetune)

| Tập | Decoder | CER (%) | Word Acc (%) | ms/ảnh |
|---|---|---:|---:|---:|
| vi_test | greedy | 95.60 | 0.00 | 23.00 |
| vi_test | beam5 | 160.30 | 0.00 | 79.24 |
| vi_test | beam20 | 159.92 | 0.00 | 386.22 |
| vi_hard | greedy | 94.92 | 0.00 | 23.54 |
| vi_hard | beam5 | 134.75 | 0.00 | 73.35 |
| vi_hard | beam20 | 134.63 | 0.00 | 337.99 |

Chi tiết: S/D/I tính theo % số ký tự nhãn (CER = S + D + I); OOV = % mẫu có ký tự ngoài charset của mô hình.

| Tập | Decoder | N mẫu | S (%) | D (%) | I (%) | OOV (%) | ms model | ms decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vi_test | greedy | 80 | 16.46 | 79.15 | 0.00 | 0.0 | 22.94 | 0.06 |
| vi_test | beam5 | 80 | 93.72 | 0.00 | 66.58 | 0.0 | 22.94 | 56.30 |
| vi_test | beam20 | 80 | 93.72 | 0.00 | 66.21 | 0.0 | 22.94 | 363.28 |
| vi_hard | greedy | 80 | 17.02 | 77.90 | 0.00 | 0.0 | 23.47 | 0.07 |
| vi_hard | beam5 | 80 | 93.03 | 0.24 | 41.49 | 0.0 | 23.47 | 49.88 |
| vi_hard | beam20 | 80 | 93.03 | 0.24 | 41.37 | 0.0 | 23.47 | 314.52 |
