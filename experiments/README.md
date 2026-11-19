# Experiments

Mỗi notebook ở đây kiểm tra **một câu hỏi** duy nhất rút ra từ bài review. Code pipeline chính vẫn giữ trong `notebooks/`.

## Phân công 4 thành viên & Bảng thực nghiệm

**Notebook tổng hợp cho cả 4 phần:** [R00_tong_hop_thuc_nghiem.ipynb](R00_tong_hop_thuc_nghiem.ipynb)

| Thành viên | Mã | Câu hỏi & Nội dung | Trạng thái |
|---|---|---|---|
| **Tổng hợp** | **R00** | Bảng & biểu đồ tổng hợp 4 phần + cấu hình đề xuất | Đã chạy |
| **Người 1** | **E06** | Trùng lặp, câu gần trùng (`near-dup`) & bắt 4–5★ khiếu nại ngầm | Đã chạy (CPU) |
| **Người 2** | **E03** | Ablation `clean_text` & tách từ `pyvi` + audit mẫu đổi sai | Đã chạy (CPU) |
| **Người 2** | **E07** | Leave-one-rule-out từng luật + Phủ định (`NOT_`) & vế sau `nhưng` | Đã chạy (CPU) |
| **Người 3** | **E01** | So sánh nhãn cảm xúc 3 lớp vs 4 lớp (`Mixed`) | Đã chạy (CPU) |
| **Người 3** | **E08** | Luật xử lý câu hoà trong 3 lớp (`tie_neutral`, `tie_negative`, `drop_ties`) | Đã chạy (CPU) |
| **Người 3** | **E09** | Cân bằng lớp (`balanced`, `smooth`, `oversample`) & ngưỡng riêng 5 khía cạnh | Đã chạy (CPU) |
| **Người 4** | **E04** | So sánh bộ tách từ `pyvi` vs `RDRSegmenter` cho PhoBERT | Đã chạy (Colab GPU) |
| **Người 4** | **E05** | PhoBERT ablation: `MAX_LEN` (128 vs 256) và `LR` trên 1 tập dev | Đã chạy (Colab GPU) |
| **Người 4** | **E10** | Augmentation lớp Trung lập (`Swap`, `Deletion`, `Reverse Teencode`) | Đã chạy (CPU) |
| *(Tham chiếu)* | **E02** | Ablation TF-IDF (`ngram`, `min_df`, mở rộng `C`) + Learning Curve | Đã chạy (CPU) |

## Khi chạy thực nghiệm và sửa pipeline thì chạy lại thế nào?
- **Lúc đang thử nghiệm (`experiments/`):** KHÔNG chạy lại từ đầu `00 -> 06`. Mỗi file `E0x` đọc thẳng `data/processed/*_clean.csv` (hoặc `src/`), đổi đúng 1 thứ và ghi kết quả vào `reports/experiments_log.csv`.
- **Sau khi chọn được cấu hình thắng để đưa vào pipeline chính (`notebooks/`):** chỉ chạy lại **từ bước bạn sửa trở về sau**.
- **Lúc nộp bài cuối cùng:** chạy sạch **1 lần duy nhất end-to-end `00 -> 06`** để đảm bảo kết quả tái lập 100%.
