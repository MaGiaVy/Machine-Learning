# Phân loại Đánh giá Khách hàng TMĐT Tiếng Việt (UIT-ViSFD)

Dự án phân loại đồng thời **Cảm xúc 3 lớp (`sentiment`)** và **Đa khía cạnh 5 nhóm (`asp_0..asp_4`)** trên dữ liệu đánh giá khách hàng tiếng Việt (UIT-ViSFD), kết hợp giải thích SHAP và đọc/dự đoán file Excel thực tế.

---

## Tổng hợp Thực nghiệm Tiền xử lý (Phân công 4 thành viên)

> **Xem báo cáo tổng hợp biểu đồ & số liệu tại 1 file duy nhất:** [experiments/R00_tong_hop_thuc_nghiem.ipynb](experiments/R00_tong_hop_thuc_nghiem.ipynb)  
> **Bảng log toàn bộ 73 lần chạy (`E01–E10`):** [reports/experiments_log.csv](reports/experiments_log.csv)

| Thành viên | Mảng tiền xử lý & bài toán TMĐT | Notebook thực nghiệm | Phát hiện chính & Hướng giải quyết |
|---|---|---|---|
| **Thành viên 1** | Chất lượng dữ liệu, câu gần trùng & **bắt 4–5★ khiếu nại ngầm** | [E06_dedup_near_dup_and_star_conflict.ipynb](experiments/E06_dedup_near_dup_and_star_conflict.ipynb) | Lọc near-dup `cosine >= 0.90` (chặn leakage); dùng text-only + `P(neg) >= 0.50` gắn cờ `SILENT_COMPLAINT` bắt `30/54` (`7.70%`) ca 4–5★ chê ngầm (Precision `71.43%`). |
| **Thành viên 2** | Ablation từng luật chuẩn hoá, phủ định & **vế đối lập sau `nhưng`** | [E03](experiments/E03_cleaning_and_teencode_ablation.ipynb) · [E07](experiments/E07_normalization_negation_and_but_clause.ipynb) | Cả 6 nhóm luật trong `src/text_cleaner.py` đều có đóng góp dương (`multi_teencode` `-0.0103`); gắn thẻ `NOT_`/`BUT_` làm thưa TF-IDF nên giữ văn bản tự nhiên và chuyển câu chuyển ý sang PhoBERT. |
| **Thành viên 3** | Thiết kế nhãn câu hoà & xử lý mất cân bằng lớp / ngưỡng | [E01](experiments/E01_label_scheme_3vs4_class.ipynb) · [E08](experiments/E08_tie_break_labeling.ipynb) · [E09](experiments/E09_class_imbalance.ipynb) | Câu hoà (`16.2%` train) là nguồn nhiễu chính của lớp Trung lập; tune **5 ngưỡng riêng trên OOF train** nâng Multi-label Aspect F1 từ `0.8911 → 0.9047`; `oversample_neu` đạt `0.7521`. |
| **Thành viên 4** | Tăng cường dữ liệu, bộ tách từ & độ dài câu PhoBERT | [E04](experiments/E04_segmenter_pyvi_vs_rdrseg.ipynb) · [E05](experiments/E05_phobert_maxlen_lr_epochs.ipynb) · [E10](experiments/E10_data_augmentation.ipynb) | Giữ `pyvi` (chênh với `RDRSegmenter` chỉ `0.0011` Mean F1); PhoBERT `MAX_LEN=128 + LR=3e-5` đạt **`0.8597`**; `Random Swap x1.0` đạt `0.7535` (tránh `Reverse Teencode`). |

---

## Cấu trúc thư mục

- `notebooks/`: Pipeline chính end-to-end (`00_load_and_merge_raw.ipynb` → `06_excel_auto_detector.ipynb`).
- `experiments/`: Toàn bộ 10 notebook thực nghiệm (`E01–E10`) và notebook tổng hợp (`R00`).
- `src/`: Mã nguồn dùng chung ([labels.py](src/labels.py), [text_cleaner.py](src/text_cleaner.py), [modeling.py](src/modeling.py), [excel_ingestion.py](src/excel_ingestion.py), [exp_log.py](src/exp_log.py)).
- `tests/`: Bộ kiểm thử đơn vị (`python -m pytest tests -q`).
- [RULE.md](RULE.md) & [coding_style.md](coding_style.md): Quy chuẩn tương thích 3 môi trường (Local / Google Colab / Kaggle GPU) và quy tắc code của nhóm.
