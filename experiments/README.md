# Experiments

Mỗi notebook ở đây kiểm tra **một câu hỏi** duy nhất rút ra từ bài review. Code pipeline chính vẫn giữ trong `notebooks/`.

## Khi chạy thực nghiệm và sửa pipeline thì chạy lại thế nào?
- **Lúc đang thử nghiệm (`experiments/`):** KHÔNG chạy lại từ đầu `00 -> 06`. Mỗi file `E0x` đọc thẳng `data/processed/*_clean.csv` (hoặc `src/`), đổi đúng 1 thứ và ghi kết quả vào `reports/experiments_log.csv`.
- **Sau khi chọn được cấu hình thắng để đưa vào pipeline chính (`notebooks/`):** chỉ chạy lại **từ bước bạn sửa trở về sau**:
  - Sửa logic nhãn (`src/labels.py` / `00`): chạy lại `00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06`.
  - Sửa làm sạch / tách từ (`src/text_cleaner.py` / `01`): chạy lại `01 -> 02 -> 03 -> 04 -> 05 -> 06` (bỏ qua `00`).
  - Chỉ sửa tham số TF-IDF / PhoBERT (`03` hoặc `04`): chỉ chạy lại `03` hoặc `04`, rồi chạy `05`/`06` nếu chúng dùng model vừa lưu.
- **Lúc nộp bài cuối cùng:** chạy sạch **1 lần duy nhất end-to-end `00 -> 06`** để đảm bảo kết quả tái lập 100%.

## Danh sách thực nghiệm
| Mã | Câu hỏi (từ review) | Trạng thái |
|---|---|---|
| E01 | Sentiment 3 lớp vs 4 lớp (Mixed) | đã có notebook & đã chạy |
| E02 | Ablation TF-IDF (ngram, min_df, max_features, mở rộng C) + Learning Curve | đã có notebook |
| E03 | Ablation làm sạch văn bản, teencode, tách từ `pyvi` + audit mẫu đổi sai | đã có notebook |
| E04 | So sánh bộ tách từ `pyvi` vs `RDRSegmenter` cho PhoBERT | đã có notebook (cần `py_vncorenlp` + GPU) |
| E05 | PhoBERT ablation: `MAX_LEN` (128 vs 256), `LR` trên 1 tập dev thống nhất | đã có notebook (cần GPU) |
