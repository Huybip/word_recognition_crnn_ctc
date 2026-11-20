# Phân tích lỗi: mid_vi / vi_test_greedy

## 1. Tổng quan

| Chỉ số | Giá trị |
|---|---|
| Số mẫu / số ký tự nhãn | 800 / 8388 |
| CER | 7.55% (95% CI: 6.47-8.54%) |
| Word Accuracy (khớp hoàn toàn) | 63.38% (95% CI: 60.25-66.75%) |
| Thay thế S / Xóa D / Chèn I | 524 / 95 / 14  (6.25% / 1.13% / 0.17% số ký tự nhãn) |

Khoảng tin cậy bootstrap theo mẫu (1000 lần). CER = (S+D+I)/N gộp toàn tập.

## 2. Phân bố lỗi theo loại

![](figures/error_types.png)

| Loại lỗi | Nhóm | Số lỗi | % tổng lỗi |
|---|---|---|---|
| Thay thế khác | Thay thế khác | 201 | 31.8 |
| Sai dấu thanh (có dấu nhưng sai loại) | Lỗi dấu tiếng Việt | 81 | 12.8 |
| Mất ký tự | Mất ký tự | 61 | 9.6 |
| Thiếu dấu thanh | Lỗi dấu tiếng Việt | 60 | 9.5 |
| Thừa dấu thanh | Lỗi dấu tiếng Việt | 37 | 5.8 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | Lỗi dấu tiếng Việt | 36 | 5.7 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | Lỗi dấu tiếng Việt | 33 | 5.2 |
| Sai cả dấu phụ lẫn dấu thanh | Lỗi dấu tiếng Việt | 29 | 4.6 |
| Sai hoa/thường (c/C, o/O...) | Hoa/thường | 26 | 4.1 |
| Mất hẳn chữ có dấu | Mất ký tự | 22 | 3.5 |
| Thừa ký tự | Thừa ký tự | 11 | 1.7 |
| Khoảng trắng bị thay | Khoảng trắng | 11 | 1.7 |
| Ký tự giống nhau (O/0, l/1/I, S/5...) | Ký tự giống nhau | 10 | 1.6 |
| Mất khoảng trắng | Khoảng trắng | 10 | 1.6 |
| Sai dấu phụ (mũ/móc/trăng nhầm nhau) | Lỗi dấu tiếng Việt | 4 | 0.6 |
| Thừa khoảng trắng | Khoảng trắng | 1 | 0.2 |

Theo nhóm lớn: Lỗi dấu tiếng Việt 280 (44.2%); Thay thế khác 201 (31.8%); Mất ký tự 83 (13.1%); Hoa/thường 26 (4.1%); Khoảng trắng 22 (3.5%); Thừa ký tự 11 (1.7%); Ký tự giống nhau 10 (1.6%)

## 3. Ma trận nhầm lẫn và top-20 cặp nhầm lẫn

![](figures/confusion_matrix.png)

Cặp ký tự (nhãn -> dự đoán; ∅ = bị xóa hoặc thừa):

| Cặp | Số lần | % tổng lỗi | % lần xuất hiện của ký tự nhãn |
|---|---|---|---|
| `␣` -> `∅` | 10 | 1.6 | 0.8 |
| `n` -> `∅` | 7 | 1.1 | 1.1 |
| `ỉ` -> `i` | 4 | 0.6 | 40.0 |
| `t` -> `∅` | 4 | 0.6 | 1.2 |
| `ỗ` -> `ổ` | 4 | 0.6 | 80.0 |
| `đ` -> `d` | 4 | 0.6 | 4.3 |
| `i` -> `t` | 4 | 0.6 | 1.6 |
| `i` -> `í` | 4 | 0.6 | 1.6 |
| `ể` -> `ế` | 4 | 0.6 | 25.0 |
| `ế` -> `ê` | 4 | 0.6 | 10.8 |
| `g` -> `∅` | 4 | 0.6 | 1.4 |
| `ầ` -> `â` | 4 | 0.6 | 17.4 |
| `Ả` -> `Á` | 4 | 0.6 | 36.4 |
| `s` -> `S` | 3 | 0.5 | 6.1 |
| `i` -> `ỉ` | 3 | 0.5 | 1.2 |
| `l` -> `∅` | 3 | 0.5 | 3.7 |
| `ổ` -> `ố` | 3 | 0.5 | 20.0 |
| `ả` -> `á` | 3 | 0.5 | 7.1 |
| `ề` -> `ể` | 3 | 0.5 | 15.0 |
| `i` -> `∅` | 3 | 0.5 | 1.2 |

Cụm lỗi liền nhau (cho thấy cả các nhầm lẫn nhiều ký tự như m -> rn):

| Nhãn | Dự đoán | Số lần |
|---|---|---|
| `ỉ` | `i` | 4 |
| `ỗ` | `ổ` | 4 |
| `i` | `í` | 4 |
| `ể` | `ế` | 4 |
| `ế` | `ê` | 4 |
| `Ả` | `Á` | 4 |
| `i` | `ỉ` | 3 |
| `ổ` | `ố` | 3 |
| `ả` | `á` | 3 |
| `đ` | `d` | 3 |
| `ề` | `ể` | 3 |
| `È` | `Ề` | 3 |

## 4. Nhóm lỗi "ký tự giống nhau"

Nhóm này chiếm 10 lỗi (1.6% tổng lỗi). Nhóm đang xét (sửa được ở `LOOKALIKE_GROUPS`): 0Oo, 1lI, 5Ss, 8B, 2Zz, 6G, 9gq, uv, il, mn, ec.

| Cặp | Số lần nhầm | Số lần ký tự nhãn xuất hiện |
|---|---|---|
| `O` -> `0` | 0 | 45 |
| `0` -> `O` | 0 | 47 |
| `o` -> `0` | 0 | 123 |
| `0` -> `o` | 1 | 47 |
| `l` -> `1` | 0 | 82 |
| `1` -> `l` | 0 | 46 |
| `l` -> `I` | 2 | 82 |
| `I` -> `l` | 0 | 60 |
| `I` -> `1` | 0 | 60 |
| `1` -> `I` | 0 | 46 |
| `S` -> `5` | 0 | 27 |
| `5` -> `S` | 0 | 35 |
| `s` -> `5` | 0 | 49 |
| `5` -> `s` | 0 | 35 |

| Cặp nhiều ký tự | Số lần |
|---|---|
| `cl` -> `d` | 0 |
| `d` -> `cl` | 0 |
| `h` -> `ln` | 0 |
| `ii` -> `u` | 0 |
| `ln` -> `h` | 0 |
| `m` -> `nn` | 0 |
| `m` -> `rn` | 0 |
| `nn` -> `m` | 0 |
| `rn` -> `m` | 0 |
| `u` -> `ii` | 0 |
| `vv` -> `w` | 0 |
| `w` -> `vv` | 0 |

## 5. Lỗi dấu tiếng Việt

Nhãn có 1872 ký tự mang dấu (22.3% số ký tự). Phân loại bằng NFD:

|  | Sai loại | Thiếu | Thừa |
|---|---|---|---|
| Dấu thanh (sắc/huyền/hỏi/ngã/nặng) | 81 | 60 | 37 |
| Dấu phụ (mũ/móc/trăng/gạch đ) | 4 | 36 | 33 |

Sai cả hai loại dấu cùng lúc: 29; mất hẳn chữ có dấu: 22.

![](figures/tone_confusion.png)

| Loại | Dấu phụ nhãn | Dấu phụ dự đoán | Số lần |
|---|---|---|---|
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | mũ | ∅ | 16 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | móc | ∅ | 13 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | mũ | 13 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | móc | 13 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | gạch | ∅ | 5 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | trăng | 4 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | gạch | 3 |
| Sai dấu phụ (mũ/móc/trăng nhầm nhau) | trăng | mũ | 2 |

### CER có dấu so với bỏ dấu

| Cách tính | CER (%) | Word Acc (%) | Giảm so với đầy đủ (điểm %) |
|---|---|---|---|
| đầy đủ | 7.55 | 63.38 | 0.00 |
| bỏ dấu | 4.18 | 82.75 | 3.36 |
| bỏ hoa/thường | 7.20 | 64.25 | 0.35 |
| bỏ dấu + hoa/thường | 3.83 | 84.12 | 3.72 |

Bỏ dấu khỏi cả nhãn lẫn dự đoán làm CER giảm 3.36 điểm (44.5% của CER), Word Acc tăng 19.37 điểm. Đây là **ước lượng** phần lỗi do dấu: bỏ dấu cũng làm thay đổi các lỗi khác (vd. đ -> d), nên con số không tách bạch tuyệt đối.

![](figures/cer_ladder.png)

## 6. Lỗi theo đặc trưng của mẫu

Mỗi bảng ghi rõ số mẫu `n`; nhóm nhỏ (n < 30) có độ nhiễu cao.

### Theo độ dài nhãn (số ký tự)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| 1-5 | 72 | 14.04 | 68.06 |
| 6-10 | 451 | 6.21 | 71.40 |
| 11-15 | 123 | 5.90 | 60.16 |
| 16-20 | 117 | 9.34 | 43.59 |
| 21+ | 37 | 9.88 | 29.73 |

![](figures/by_length.png)

### Theo kiểu chữ

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| có chữ số/mã | 55 | 11.97 | 63.64 |
| CHỮ HOA | 158 | 10.52 | 45.57 |
| Viết Hoa Từng Từ | 82 | 6.80 | 62.20 |
| Hoa đầu câu/hỗn hợp | 127 | 6.54 | 64.57 |
| chữ thường | 378 | 6.29 | 70.63 |

![](figures/by_case.png)

### Theo T/(L+R) (T = rộng@H=32 chia 4; càng thấp càng sát giới hạn CTC)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| 1.5-2 | 22 | 10.34 | 36.36 |
| 2-3 | 351 | 7.50 | 61.54 |
| >=3 | 427 | 7.34 | 66.28 |

![](figures/by_ratio.png)

### Theo độ tương phản ảnh (độ lệch chuẩn điểm ảnh)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| Q1 (tương phản thấp nhất) | 200 | 13.27 | 50.50 |
| Q2 | 200 | 8.25 | 58.50 |
| Q3 | 200 | 4.75 | 69.50 |
| Q4 (cao nhất) | 200 | 3.51 | 75.00 |

![](figures/by_contrast.png)

### Theo họ font (chỉ họ có đủ mẫu, 10 họ tệ nhất)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| DroidSans | 18 | 17.50 | 66.67 |
| DejaVuSans | 46 | 9.94 | 63.04 |
| NotoSerifCJK | 66 | 9.86 | 65.15 |
| DejaVuSansCondensed | 44 | 8.83 | 52.27 |
| FreeSans | 51 | 8.45 | 54.90 |
| FreeMono | 46 | 8.05 | 60.87 |
| LiberationMono | 32 | 7.94 | 62.50 |
| RobotoCondensed | 65 | 7.79 | 66.15 |
| OpenSans | 112 | 7.76 | 60.71 |
| LiberationSerif | 46 | 6.89 | 71.74 |

![](figures/by_font.png)

## 7. Ví dụ dự đoán

Dự đoán đúng được chọn ngẫu nhiên có seed; dự đoán sai được chọn lần lượt từ loại lỗi phổ biến nhất, ưu tiên mẫu ít lỗi cho dễ đọc (không chọn theo cảm tính). Cột nguyên nhân là **giả thuyết** suy từ loại lỗi và các đặc trưng đo được của ảnh, chưa kiểm chứng bằng thí nghiệm.

### 10 dự đoán đúng

![](figures/examples_correct.png)

| # | Ảnh | Nhãn = dự đoán | Font |
|---|---|---|---|
| 1 | ![](examples/correct_01.png) | `TÂN KHÁNH HOÀ` | RobotoCondensed-Regular.ttf |
| 2 | ![](examples/correct_02.png) | `Hiện Tại` | FreeSans.ttf |
| 3 | ![](examples/correct_03.png) | `Nghiên Cứu` | FreeSerifBoldItalic.ttf |
| 4 | ![](examples/correct_04.png) | `7875700153` | DejaVuSerif-Bold.ttf |
| 5 | ![](examples/correct_05.png) | `suối giàng` | Roboto-BoldItalic.ttf |
| 6 | ![](examples/correct_06.png) | `Năm` | FreeSerifBoldItalic.ttf |
| 7 | ![](examples/correct_07.png) | `Loại Bỏ` | DejaVuSans-ExtraLight.ttf |
| 8 | ![](examples/correct_08.png) | `phón g đại loá mắt` | FreeMonoBoldOblique.ttf |
| 9 | ![](examples/correct_09.png) | `Dấu Giọng` | NotoSerifCJK-Bold.ttc |
| 10 | ![](examples/correct_10.png) | `XÔ XÁT` | LiberationSerif-Bold.ttf |

### 10 dự đoán sai

![](figures/examples_wrong.png)

| # | Ảnh | Nhãn | Dự đoán | Chỗ sai | Loại lỗi | Nhận xét nguyên nhân |
|---|---|---|---|---|---|---|
| 1 | ![](examples/wrong_01.png) | `bôn cạnh` | `bền cạnh` | `ô→ề` | Thay thế khác | thay thế khác: nhầm hai ký tự không thuộc nhóm nào ở trên | bối cảnh đo được: font OpenSans-Bold.ttf |
| 2 | ![](examples/wrong_02.png) | `tự cấp xô đẩy` | `tự cấp xô đấy` | `ẩ→ấ` | Sai dấu thanh (có dấu nhưng sai loại) | sai dấu thanh (có dấu nhưng sai loại): dấu thanh là nét nhỏ, dễ bị nhầm loại khi ảnh bị thu nhỏ hoặc nhiễu | bối cảnh đo được: font LiberationSans-BoldItalic.ttf |
| 3 | ![](examples/wrong_03.png) | `ĐẦU VAN` | `ĐẦU VN` | `A→∅` | Mất ký tự | mất ký tự: ký tự bị bỏ sót: T hẹp so với nhãn, nét mờ hoặc mô hình chưa hội tụ | bối cảnh đo được: ảnh thuộc nhóm tương phản thấp nhất 25%, font Roboto-Light.ttf |
| 4 | ![](examples/wrong_04.png) | `Đỉnh Tròn` | `Đinh Tròn` | `ỉ→i` | Thiếu dấu thanh | thiếu dấu thanh: dấu thanh mảnh bị mờ/mất sau khi resize hoặc bị nền nhiễu che | bối cảnh đo được: ảnh thuộc nhóm tương phản thấp nhất 25%, font Carlito-Regular.ttf |
| 5 | ![](examples/wrong_05.png) | `anh linh` | `anh lính` | `i→í` | Thừa dấu thanh | thừa dấu thanh: nhiễu/nét nền bị đọc thành dấu thanh | bối cảnh đo được: font FreeSerifBold.ttf |
| 6 | ![](examples/wrong_06.png) | `ÓC ĐẬU` | `ÓC ĐẠU` | `Ậ→Ạ` | Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc): dấu mũ/móc/trăng nhỏ, gần như trùng hình với chữ gốc ở độ phân giải thấp | bối cảnh đo được: font Carlito-Bold.ttf |
| 7 | ![](examples/wrong_07.png) | `hoan lạc phụt` | `hoan lạc phựt` | `ụ→ự` | Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ): nét nhỏ trên chữ bị đọc thành dấu phụ | bối cảnh đo được: font LiberationMono-Bold.ttf |
| 8 | ![](examples/wrong_08.png) | `Nhường Ngôi` | `Nhường Ngõi` | `ô→õ` | Sai cả dấu phụ lẫn dấu thanh | sai cả dấu phụ lẫn dấu thanh: chữ có nhiều dấu chồng nhau, cả hai dấu đều bị mất hoặc sai | bối cảnh đo được: font RobotoCondensed-BoldItalic.ttf |
| 9 | ![](examples/wrong_09.png) | `siêu thọ` | `Siêu thọ` | `s→S` | Sai hoa/thường (c/C, o/O...) | sai hoa/thường (c/c, o/o...): chữ hoa và thường cùng hình dạng, chỉ khác kích thước | bối cảnh đo được: ảnh thuộc nhóm tương phản thấp nhất 25%, font LiberationMono-Regular.ttf |
| 10 | ![](examples/wrong_10.png) | `TRẺ HOÁ LAI MÁU` | `TR HOÁ LAI MÁU` | `Ẻ→∅` | Mất hẳn chữ có dấu | mất hẳn chữ có dấu: chữ có dấu bị bỏ hẳn, thường do T hẹp, ảnh mờ hoặc mô hình thiên về blank | bối cảnh đo được: font RobotoCondensed-LightItalic.ttf |

## 8. Đề xuất cải thiện (rút ra từ số liệu ở trên)

Mỗi đề xuất kèm con số quan sát được. Đây là tương quan từ một lần đánh giá, chưa phải bằng chứng nhân quả: cách kiểm chứng là thay đổi đúng một yếu tố và chạy lại `eval.py` + phân tích này.

1. **Lỗi dấu chiếm 44.2% số lỗi** (dấu thanh 178, dấu phụ 73, cả hai 29). Bỏ dấu khỏi cả nhãn lẫn dự đoán làm CER giảm 3.4 điểm phần trăm (7.5% -> 4.2%). Chữ có dấu bị lỗi nhiều nhất: 'ế' (13), 'ô' (11), 'ấ' (10), 'ồ' (9), 'ầ' (9). Đề xuất: (1) tăng `img_height` lên 48 hoặc 64 để nét dấu rõ hơn (ở H=32 dấu chỉ còn khoảng 3 px); (2) sinh thêm dữ liệu có nhiều chữ hoa có dấu chồng và các chữ nêu trên; (3) dùng beam search kèm từ điển tiếng Việt để sửa dấu (chỉ có tác dụng nếu CER không dấu đã thấp).
2. **Ảnh tương phản thấp khó hơn rõ rệt**: CER 13.3% (25% ảnh tương phản thấp nhất) so với 3.5% (25% cao nhất). Thử chuẩn hóa tương phản từng ảnh (vd. CLAHE) hoặc tăng augmentation tương phản.

## 9. Phương pháp và giới hạn

- Căn chỉnh bằng Levenshtein DP (khớp/thay > xóa > chèn khi hòa); căn chỉnh tối thiểu không duy nhất nên cặp nhầm lẫn phụ thuộc quy tắc hòa.
- Mỗi thao tác lỗi thuộc đúng một loại; cụm lỗi khớp m<->rn, cl<->d... được gán 'cặp giống nhau' cho toàn cụm.
- Chữ gốc và dấu tách bằng `unicodedata` NFD; 'đ' xử lý riêng (gạch ngang). Khi cả dấu phụ và dấu thanh cùng sai thì xếp vào loại 'sai cả hai'.
- Họ font suy ra từ tên file; T/(L+R) tính với `--img_height` và `--stride` đã khai báo (mặc định 32 và 4).
