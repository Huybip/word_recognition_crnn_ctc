# Phân tích lỗi: mid_vi / vi_hard_greedy

> **Lưu ý:** CER tập này (50.5%) cao gấp 6.8 lần val CER của cùng mô hình (7.4%): mô hình đã hội tụ trên phân phối huấn luyện, còn tập này khó hoặc khác phân phối.

## 1. Tổng quan

| Chỉ số | Giá trị |
|---|---|
| Số mẫu / số ký tự nhãn | 400 / 4167 |
| CER | 50.49% (95% CI: 47.02-54.10%) |
| Word Accuracy (khớp hoàn toàn) | 13.25% (95% CI: 10.00-16.75%) |
| Thay thế S / Xóa D / Chèn I | 1328 / 704 / 72  (31.87% / 16.89% / 1.73% số ký tự nhãn) |

Khoảng tin cậy bootstrap theo mẫu (1000 lần). CER = (S+D+I)/N gộp toàn tập.

## 2. Phân bố lỗi theo loại

![](figures/error_types.png)

| Loại lỗi | Nhóm | Số lỗi | % tổng lỗi |
|---|---|---|---|
| Thay thế khác | Thay thế khác | 978 | 46.5 |
| Mất ký tự | Mất ký tự | 457 | 21.7 |
| Mất hẳn chữ có dấu | Mất ký tự | 155 | 7.4 |
| Khoảng trắng bị thay | Khoảng trắng | 92 | 4.4 |
| Mất khoảng trắng | Khoảng trắng | 74 | 3.5 |
| Sai cả dấu phụ lẫn dấu thanh | Lỗi dấu tiếng Việt | 73 | 3.5 |
| Thừa ký tự | Thừa ký tự | 49 | 2.3 |
| Thiếu dấu thanh | Lỗi dấu tiếng Việt | 45 | 2.1 |
| Sai dấu thanh (có dấu nhưng sai loại) | Lỗi dấu tiếng Việt | 38 | 1.8 |
| Thừa dấu thanh | Lỗi dấu tiếng Việt | 38 | 1.8 |
| Sai hoa/thường (c/C, o/O...) | Hoa/thường | 31 | 1.5 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | Lỗi dấu tiếng Việt | 22 | 1.0 |
| Ký tự giống nhau (O/0, l/1/I, S/5...) | Ký tự giống nhau | 21 | 1.0 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | Lỗi dấu tiếng Việt | 16 | 0.8 |
| Sai dấu phụ (mũ/móc/trăng nhầm nhau) | Lỗi dấu tiếng Việt | 10 | 0.5 |
| Thừa khoảng trắng | Khoảng trắng | 5 | 0.2 |

Theo nhóm lớn: Thay thế khác 978 (46.5%); Mất ký tự 612 (29.1%); Lỗi dấu tiếng Việt 242 (11.5%); Khoảng trắng 171 (8.1%); Thừa ký tự 49 (2.3%); Hoa/thường 31 (1.5%); Ký tự giống nhau 21 (1.0%)

## 3. Ma trận nhầm lẫn và top-20 cặp nhầm lẫn

![](figures/confusion_matrix.png)

Cặp ký tự (nhãn -> dự đoán; ∅ = bị xóa hoặc thừa):

| Cặp | Số lần | % tổng lỗi | % lần xuất hiện của ký tự nhãn |
|---|---|---|---|
| `␣` -> `∅` | 74 | 3.5 | 12.1 |
| `h` -> `∅` | 45 | 2.1 | 18.5 |
| `n` -> `∅` | 34 | 1.6 | 11.8 |
| `g` -> `∅` | 28 | 1.3 | 20.6 |
| `t` -> `∅` | 25 | 1.2 | 18.4 |
| `c` -> `∅` | 18 | 0.9 | 16.7 |
| `i` -> `∅` | 17 | 0.8 | 11.4 |
| `H` -> `∅` | 16 | 0.8 | 18.6 |
| `u` -> `∅` | 14 | 0.7 | 14.9 |
| `m` -> `∅` | 14 | 0.7 | 21.2 |
| `I` -> `∅` | 14 | 0.7 | 26.9 |
| `b` -> `∅` | 13 | 0.6 | 29.5 |
| `a` -> `∅` | 13 | 0.6 | 20.6 |
| `l` -> `∅` | 12 | 0.6 | 27.3 |
| `o` -> `∅` | 12 | 0.6 | 17.1 |
| `N` -> `∅` | 11 | 0.5 | 12.1 |
| `m` -> `n` | 11 | 0.5 | 16.7 |
| `p` -> `∅` | 10 | 0.5 | 16.9 |
| `k` -> `∅` | 9 | 0.4 | 32.1 |
| `ạ` -> `∅` | 9 | 0.4 | 22.5 |

Cụm lỗi liền nhau (cho thấy cả các nhầm lẫn nhiều ký tự như m -> rn):

| Nhãn | Dự đoán | Số lần |
|---|---|---|
| `g` | `∅` | 6 |
| `í` | `i` | 4 |
| `c` | `t` | 4 |
| `h` | `∅` | 4 |
| `t` | `∅` | 3 |
| `d` | `đ` | 3 |
| `ỉ` | `i` | 3 |
| `ê` | `ế` | 3 |
| `ấ` | `á` | 3 |
| `n` | `∅` | 3 |
| `ặ` | `ậ` | 2 |
| `ó` | `ỏ` | 2 |

## 4. Nhóm lỗi "ký tự giống nhau"

Nhóm này chiếm 21 lỗi (1.0% tổng lỗi). Nhóm đang xét (sửa được ở `LOOKALIKE_GROUPS`): 0Oo, 1lI, 5Ss, 8B, 2Zz, 6G, 9gq, uv, il, mn, ec.

| Cặp | Số lần nhầm | Số lần ký tự nhãn xuất hiện |
|---|---|---|
| `O` -> `0` | 0 | 15 |
| `0` -> `O` | 0 | 23 |
| `o` -> `0` | 0 | 70 |
| `0` -> `o` | 1 | 23 |
| `l` -> `1` | 0 | 44 |
| `1` -> `l` | 0 | 29 |
| `l` -> `I` | 0 | 44 |
| `I` -> `l` | 0 | 52 |
| `I` -> `1` | 0 | 52 |
| `1` -> `I` | 1 | 29 |
| `S` -> `5` | 0 | 19 |
| `5` -> `S` | 1 | 17 |
| `s` -> `5` | 0 | 26 |
| `5` -> `s` | 0 | 17 |

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

Nhãn có 925 ký tự mang dấu (22.2% số ký tự). Phân loại bằng NFD:

|  | Sai loại | Thiếu | Thừa |
|---|---|---|---|
| Dấu thanh (sắc/huyền/hỏi/ngã/nặng) | 38 | 45 | 38 |
| Dấu phụ (mũ/móc/trăng/gạch đ) | 10 | 22 | 16 |

Sai cả hai loại dấu cùng lúc: 73; mất hẳn chữ có dấu: 155.

![](figures/tone_confusion.png)

| Loại | Dấu phụ nhãn | Dấu phụ dự đoán | Số lần |
|---|---|---|---|
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | mũ | ∅ | 10 |
| Sai dấu phụ (mũ/móc/trăng nhầm nhau) | mũ | móc | 6 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | gạch | ∅ | 6 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | mũ | 5 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | gạch | 5 |
| Sai dấu phụ (mũ/móc/trăng nhầm nhau) | trăng | mũ | 4 |
| Thiếu dấu phụ (â/ă/ê/ô/ơ/ư/đ -> chữ gốc) | trăng | ∅ | 4 |
| Thừa dấu phụ (chữ gốc -> â/ă/ê/ô/ơ/ư/đ) | ∅ | móc | 4 |

### CER có dấu so với bỏ dấu

| Cách tính | CER (%) | Word Acc (%) | Giảm so với đầy đủ (điểm %) |
|---|---|---|---|
| đầy đủ | 50.49 | 13.25 | 0.00 |
| bỏ dấu | 43.99 | 25.75 | 6.50 |
| bỏ hoa/thường | 49.17 | 13.50 | 1.32 |
| bỏ dấu + hoa/thường | 42.26 | 27.00 | 8.23 |

Bỏ dấu khỏi cả nhãn lẫn dự đoán làm CER giảm 6.50 điểm (12.9% của CER), Word Acc tăng 12.50 điểm. Đây là **ước lượng** phần lỗi do dấu: bỏ dấu cũng làm thay đổi các lỗi khác (vd. đ -> d), nên con số không tách bạch tuyệt đối.

![](figures/cer_ladder.png)

## 6. Lỗi theo đặc trưng của mẫu

Mỗi bảng ghi rõ số mẫu `n`; nhóm nhỏ (n < 30) có độ nhiễu cao.

### Theo độ dài nhãn (số ký tự)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| 1-5 | 42 | 52.80 | 26.19 |
| 6-10 | 213 | 46.11 | 16.43 |
| 11-15 | 68 | 50.00 | 7.35 |
| 16-20 | 64 | 52.87 | 3.12 |
| 21+ | 13 | 67.94 | 0.00 |

![](figures/by_length.png)

### Theo kiểu chữ

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| có chữ số/mã | 30 | 59.92 | 10.00 |
| Viết Hoa Từng Từ | 37 | 52.93 | 18.92 |
| CHỮ HOA | 77 | 50.86 | 7.79 |
| chữ thường | 189 | 50.27 | 14.29 |
| Hoa đầu câu/hỗn hợp | 67 | 46.00 | 14.93 |

![](figures/by_case.png)

### Theo T/(L+R) (T = rộng@H=32 chia 4; càng thấp càng sát giới hạn CTC)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| <1.5 | 2 | 88.37 | 0.00 |
| 1.5-2 | 57 | 66.41 | 0.00 |
| 2-3 | 236 | 48.06 | 12.29 |
| >=3 | 105 | 37.60 | 22.86 |

![](figures/by_ratio.png)

### Theo độ tương phản ảnh (độ lệch chuẩn điểm ảnh)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| Q1 (tương phản thấp nhất) | 100 | 70.79 | 1.00 |
| Q2 | 100 | 54.81 | 4.00 |
| Q3 | 100 | 47.10 | 17.00 |
| Q4 (cao nhất) | 100 | 25.61 | 31.00 |

![](figures/by_contrast.png)

### Theo họ font (chỉ họ có đủ mẫu, 10 họ tệ nhất)

| Nhóm | n | CER (%) | Word Acc (%) |
|---|---|---|---|
| Lora | 28 | 65.31 | 14.29 |
| DejaVuSerifCondensed | 65 | 55.49 | 4.62 |
| NotoSansCJK | 86 | 50.17 | 15.12 |
| Lato | 221 | 47.24 | 14.93 |

![](figures/by_font.png)

## 7. Ví dụ dự đoán

Dự đoán đúng được chọn ngẫu nhiên có seed; dự đoán sai được chọn lần lượt từ loại lỗi phổ biến nhất, ưu tiên mẫu ít lỗi cho dễ đọc (không chọn theo cảm tính). Cột nguyên nhân là **giả thuyết** suy từ loại lỗi và các đặc trưng đo được của ảnh, chưa kiểm chứng bằng thí nghiệm.

### 10 dự đoán đúng

![](figures/examples_correct.png)

| # | Ảnh | Nhãn = dự đoán | Font |
|---|---|---|---|
| 1 | ![](examples/correct_01.png) | `bụng nhụng` | Lora-Variable.ttf |
| 2 | ![](examples/correct_02.png) | `THÔNG BẠCH` | Lato-HeavyItalic.ttf |
| 3 | ![](examples/correct_03.png) | `nằm vật` | Lato-Semibold.ttf |
| 4 | ![](examples/correct_04.png) | `sươi` | Lato-MediumItalic.ttf |
| 5 | ![](examples/correct_05.png) | `Quy bản` | NotoSansCJK-Black.ttc |
| 6 | ![](examples/correct_06.png) | `Xốc nách` | Lora-Italic-Variable.ttf |
| 7 | ![](examples/correct_07.png) | `đậu rồng` | Lato-Heavy.ttf |
| 8 | ![](examples/correct_08.png) | `bắt nét` | Lato-Black.ttf |
| 9 | ![](examples/correct_09.png) | `ucnMxA` | Lato-HeavyItalic.ttf |
| 10 | ![](examples/correct_10.png) | `thu mình` | Lora-Italic-Variable.ttf |

### 10 dự đoán sai

![](figures/examples_wrong.png)

| # | Ảnh | Nhãn | Dự đoán | Chỗ sai | Loại lỗi | Nhận xét nguyên nhân |
|---|---|---|---|---|---|---|
| 1 | ![](examples/wrong_01.png) | `07/07/2019` | `07/0712019` | `/→1` | Thay thế khác | thay thế khác: nhầm hai ký tự không thuộc nhóm nào ở trên | bối cảnh đo được: font Lato-BlackItalic.ttf |
| 2 | ![](examples/wrong_02.png) | `89.167.658` | `89.167658` | `.→∅` | Mất ký tự | mất ký tự: ký tự bị bỏ sót: T hẹp so với nhãn, nét mờ hoặc mô hình chưa hội tụ | bối cảnh đo được: font Lato-Bold.ttf |
| 3 | ![](examples/wrong_03.png) | `sưởi` | `shi` | `ư→∅, ở→h` | Mất hẳn chữ có dấu, Thay thế khác | mất hẳn chữ có dấu: chữ có dấu bị bỏ hẳn, thường do T hẹp, ảnh mờ hoặc mô hình thiên về blank; thay thế khác: nhầm hai ký tự không thuộc nhóm nào ở trên | bối cảnh đo được: font DejaVuSerifCondensed-Bold.ttf |
| 4 | ![](examples/wrong_04.png) | `Db1Tb` | `Dhi Th` | `∅→h, b→i, 1→ , b→h` | Thừa ký tự, Thay thế khác, Khoảng trắng bị thay | thay thế khác: nhầm hai ký tự không thuộc nhóm nào ở trên; thừa ký tự: nhiễu nền hoặc nét đứt bị đọc thành ký tự thừa; khoảng trắng bị thay: khoảng trắng hẹp hoặc rộng tùy font nên ranh giới từ khó xác định | bối cảnh đo được: font Lato-SemiboldItalic.ttf |
| 5 | ![](examples/wrong_05.png) | `u mê` | `an` | `u→∅,  →∅, m→a, ê→n` | Mất ký tự, Mất khoảng trắng, Thay thế khác | thay thế khác: nhầm hai ký tự không thuộc nhóm nào ở trên; mất ký tự: ký tự bị bỏ sót: T hẹp so với nhãn, nét mờ hoặc mô hình chưa hội tụ; mất khoảng trắng: khoảng trắng hẹp bị bỏ qua | bối cảnh đo được: font NotoSansCJK-Bold.ttc |
| 6 | ![](examples/wrong_06.png) | `nhau` | `nhầu` | `a→ầ` | Sai cả dấu phụ lẫn dấu thanh | sai cả dấu phụ lẫn dấu thanh: chữ có nhiều dấu chồng nhau, cả hai dấu đều bị mất hoặc sai | bối cảnh đo được: ảnh thuộc nhóm tương phản thấp nhất 25%, font Lato-Heavy.ttf |
| 7 | ![](examples/wrong_07.png) | `hồn tinh vệ` | `hôồn tỉnh vệ` | `∅→ô, i→ỉ` | Thừa ký tự, Thừa dấu thanh | thừa ký tự: nhiễu nền hoặc nét đứt bị đọc thành ký tự thừa; thừa dấu thanh: nhiễu/nét nền bị đọc thành dấu thanh | bối cảnh đo được: font Lato-Black.ttf |
| 8 | ![](examples/wrong_08.png) | `Khí vật` | `Khi vật` | `í→i` | Thiếu dấu thanh | thiếu dấu thanh: dấu thanh mảnh bị mờ/mất sau khi resize hoặc bị nền nhiễu che | bối cảnh đo được: font Lato-BoldItalic.ttf |
| 9 | ![](examples/wrong_09.png) | `TAM BẢN CHINH CUNG` | `TAM BÀN CHINH CUNG` | `Ả→À` | Sai dấu thanh (có dấu nhưng sai loại) | sai dấu thanh (có dấu nhưng sai loại): dấu thanh là nét nhỏ, dễ bị nhầm loại khi ảnh bị thu nhỏ hoặc nhiễu | bối cảnh đo được: font Lato-Semibold.ttf |
| 10 | ![](examples/wrong_10.png) | `Tôm gọng phởn phơ` | `Tôm gọng phởn phở` | `ơ→ở` | Thừa dấu thanh | thừa dấu thanh: nhiễu/nét nền bị đọc thành dấu thanh | bối cảnh đo được: font NotoSansCJK-Black.ttc |

## 8. Đề xuất cải thiện (rút ra từ số liệu ở trên)

Mỗi đề xuất kèm con số quan sát được. Đây là tương quan từ một lần đánh giá, chưa phải bằng chứng nhân quả: cách kiểm chứng là thay đổi đúng một yếu tố và chạy lại `eval.py` + phân tích này.

1. **Ưu tiên 1: khoảng cách khái quát hóa, không phải thiếu huấn luyện.** CER = 50.5% và Word Acc = 13.2% (khoảng tin cậy CER 95%: 47.0-54.1%). Cùng mô hình đạt val CER 7.4% trên phân phối huấn luyện, tức là đã hội tụ ở đó nhưng suy giảm 6.8 lần ở tập này. Tập này khác phân phối (ví dụ font chưa thấy, nhiễu/mờ nặng) nên huấn luyện thêm trên cùng dữ liệu khó có thể giúp. Để tách nguyên nhân: sinh hai tập mới, một tập dùng font của train + augmentation nặng, một tập dùng font lạ + augmentation thường, rồi so CER; sau đó đưa một phần augmentation nặng và nhiều họ font hơn vào train.
2. **Ảnh tương phản thấp khó hơn rõ rệt**: CER 70.8% (25% ảnh tương phản thấp nhất) so với 25.6% (25% cao nhất). Thử chuẩn hóa tương phản từng ảnh (vd. CLAHE) hoặc tăng augmentation tương phản.

## 9. Phương pháp và giới hạn

- Căn chỉnh bằng Levenshtein DP (khớp/thay > xóa > chèn khi hòa); căn chỉnh tối thiểu không duy nhất nên cặp nhầm lẫn phụ thuộc quy tắc hòa.
- Mỗi thao tác lỗi thuộc đúng một loại; cụm lỗi khớp m<->rn, cl<->d... được gán 'cặp giống nhau' cho toàn cụm.
- Chữ gốc và dấu tách bằng `unicodedata` NFD; 'đ' xử lý riêng (gạch ngang). Khi cả dấu phụ và dấu thanh cùng sai thì xếp vào loại 'sai cả hai'.
- Họ font suy ra từ tên file; T/(L+R) tính với `--img_height` và `--stride` đã khai báo (mặc định 32 và 4).
