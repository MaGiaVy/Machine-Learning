# Phân loại Đánh giá Khách hàng TMĐT Tiếng Việt (UIT-ViSFD) — v4.0 True ABSA (5 × 4)

Dự án **True Aspect-Based Sentiment Analysis (`5 × 4`)** trên tập dữ liệu **đánh giá điện thoại di động tiếng Việt (UIT-ViSFD)**: dự đoán đồng thời cực tính `0=Không đề cập, 1=Tiêu cực, 2=Trung lập, 3=Tích cực` cho 5 nhóm khía cạnh (`PERF_FEAT`, `SCREEN_CAM`, `BATTERY`, `PRICE_SER`, `DESIGN_GEN`), kết hợp giải thích **SHAP + PhoBERT Occlusion XAI**, gom cụm **Actionable Insights** và bộ đọc/dự đoán Excel thực tế.

## Kết quả Đối chiếu trên Tập Test (`2,224` review; Train sạch `7,762` dòng)

| Chỉ số (Test `N=2,224`) | TF-IDF Baseline (`LinearSVC`, `C=0.5`) | PhoBERT ABSA v4.0 (`MAX_LEN=128`) | Chênh lệch |
|---|---:|---:|---:|
| **ABSA Macro F1** | `0.7556` | **`0.8417`** | **+0.0861** *(95% Paired Bootstrap CI: `[+0.0769, +0.0964]`)* |
| **ABSA Micro F1** | `0.7576` | **`0.8415`** | **+0.0839** |
| **Mean 4-class Macro F1** | `0.6895` | **`0.7929`** | **+0.1034** |
| **Polarity-given-Detected Macro F1** | `0.7854` | **`0.8370`** | **+0.0516** |
| **Tập xung đột (`has_conflict`, `n=286`)** | `0.6530` | **`0.7533`** | **+0.1003** |
| **Tập xung đột có Negative (`n=141`)** | `0.6316` | **`0.7337`** | **+0.1021** |

- **Mức độ khiếu nại ngầm trên Test (`Muc_Do_Khieu_Nai_Ngam`):** `Không: 942`, `Nhẹ: 250`, `Cao: 205` (`14.67%` trong nhóm 4–5★), `N/A: 827` (1–3★).
- **Lưu ý trọng số (Weights):** File `models/phobert_absa_v4/phobert_absa_best.pt` (~540 MB) và gói `phobert_absa_outputs_v4.zip` bị loại khỏi Git theo `.gitignore`. Để tái tạo weights, chạy [notebooks/04_phobert_multitask.ipynb](notebooks/04_phobert_multitask.ipynb) trên Google Colab T4 GPU (~8–10 phút) với gói `src_and_data_v4.zip`, sau đó giải nén `phobert_absa_outputs_v4.zip` vào `models/phobert_absa_v4/` và `data/processed/`.

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

- **Dữ liệu:** review điện thoại (UIT-ViSFD). 10 file Excel trong `data/samples_messy/` là dữ liệu **mô phỏng (synthetic)** các định dạng xuất thô từ sàn TMĐT, chỉ dùng để thử NB06.
- `notebooks/`: Pipeline chính end-to-end (`00_load_and_merge_raw.ipynb` → `06_excel_auto_detector.ipynb`).
- `experiments/`: Toàn bộ 10 notebook thực nghiệm (`E01–E10`) và notebook tổng hợp (`R00`).
- `src/`: Mã nguồn dùng chung ([labels.py](src/labels.py), [text_cleaner.py](src/text_cleaner.py), [modeling.py](src/modeling.py), [excel_ingestion.py](src/excel_ingestion.py), [exp_log.py](src/exp_log.py)).
- `tests/`: Bộ kiểm thử đơn vị (`python -m pytest tests -q`).
- [RULE.md](RULE.md) & [coding_style.md](coding_style.md): Quy chuẩn tương thích 3 môi trường (Local / Google Colab / Kaggle GPU) và quy tắc code của nhóm.


## Hạn chế đã biết

- **Neutral yếu:** F1 Neutral 0.33–0.79 theo khía cạnh (test chỉ có 78–147 mẫu Neutral/khía cạnh, trừ PRICE_SER 344), kéo `mean_4cls_macro_f1` xuống. Với bài toán khiếu nại, chỉ số chính là F1 Negative (0.71–0.88).
- **Một seed huấn luyện (42):** Paired Bootstrap CI chỉ phản ánh dao động mẫu test, không phản ánh dao động huấn luyện. Chênh lệch +0.086 đủ lớn nên kết luận vẫn vững.
- **`needs_review`:** ngưỡng theo ngân sách ~15% review; chỉ bắt được một phần lỗi (xem `model_metadata.json` -> `needs_review_test`). Không dùng điều kiện mâu thuẫn sao vì sẽ trùng với khiếu nại ngầm.
- **SHAP (NB05)** giải thích mô hình surrogate LR/TF-IDF, không phải PhoBERT.
