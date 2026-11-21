# VoC-Insight v4.1 — Kế Hoạch Chỉnh Sửa Gọn (Không Train Lại GPU)

**Goal:** Đóng các lỗi thực sự có giá trị (khớp số liệu README, kết luận phân tích lỗi NB04, từ khóa NB05 không trùng/mảnh từ, test khóa phát hành) mà **không cần train lại PhoBERT trên Colab** và **không tạo thêm file thừa**.

## Phạm vi giữ lại vs. Bỏ qua

| # | Mục | Quyết định | Lý do |
|---|---|---|---|
| 1 | README lệch số (`0.8417` vs `0.8410`), thiếu `review_exact_match`, thiếu cảnh báo `dev_pred_phobert.csv` & link weights | **Làm (Cao)** | Khớp tuyệt đối `model_metadata.json`, tránh lệch số khi bảo vệ. |
| 2 | Error analysis NB04 thiếu kết luận | **Làm (Cao)** | Thêm markdown `### Kết luận phân tích lỗi` + lấy mẫu 5 câu sai tự tin cao & 5 ngẫu nhiên (chạy từ `test_pred_phobert.csv`, không cần nạp model/GPU). |
| 3 | Từ khóa NB05 còn trùng giữa cụm & có mảnh từ (`bảo`, `triệu`) | **Làm (Trung bình)** | Gán mỗi từ khóa về đúng 1 cụm có điểm log-odds cao nhất + thêm mảnh từ vào `VN_STOP`, chạy lại NB05. |
| 4 | Khóa cũ `low_conf_threshold` trong `calibration.json` & NB06 | **Làm (Trung bình)** | Dọn khóa cũ ở NB06 sang `review_conf_threshold`; cập nhật `src/modeling.py` để lần chạy sau không ghi khóa cũ. |
| 5 | Thiếu test phát hành (`needs_review ≈ 15%`, `Hỗn hợp`, `Khia_Canh_Can_Kiem`, fingerprint, `.gitignore`) | **Làm (Trung bình)** | Gom vào `tests/test_v4_release_checks.py` + thêm `*.pt` vào `.gitignore`. |
| 6 | Near-dup bỏ ký tự `+` (cặp 6587) | **Bỏ qua** | Đã được `NEARDUP_KEEP_IDS` chặn an toàn; sửa có rủi ro đổi tập train 7,762 dòng. |
| 7 | 126 warnings từ `joblib`/`anyio` trong `pytest` | **Bỏ qua** | Warning nội bộ của thư viện ngoài với NumPy 2.5 / Pytest 9, không phải lỗi code nhóm. |
| 8 | Tạo thêm 4 file mới (`src/dedup_keys.py`, `src/analysis.py`, `scripts/pack_src_and_data.py`, `scripts/render_readme_metrics.py`) | **Bỏ qua** | Quá tay cho vài hàm ngắn; đặt thẳng hàm vào `src/labels.py` / `src/modeling.py` có sẵn. |
| 9 | Train lại NB04 trên Colab | **Bỏ qua** | Không đụng gì tới weights; train lại tốn thời gian và làm lệch số 0.001–0.003 vô ích. |

---

## Các bước thực hiện (Local CPU, không cần Colab)

### Task 1: `.gitignore` & Test phát hành (`tests/test_v4_release_checks.py`)
- [ ] Thêm `*.pt` và `.tmp_run/` vào `.gitignore` (giữ nguyên các dòng chặn `phobert/` và `*.zip` đã có).
- [ ] Tạo `tests/test_v4_release_checks.py` kiểm tra:
  - `.gitignore` chặn `*.pt`, `phobert/`, `phobert_absa_eval_v4*.zip`, `src_and_data*.zip`.
  - Kích thước `(7762, 1110, 2224)`, fingerprint train `ad1b0833a88851e2`, `dev ∩ test = ∅`.
  - `needs_review` trên `test_pred_phobert.csv` khớp `model_metadata.json` (`test_flag_rate` trong khoảng ngân sách 15% ± 3%).
  - `sample_excel_predicted.xlsx` có đủ cột `Du_Doan_Cam_Xuc` (có trạng thái `Hỗn hợp`), `Khia_Canh_Can_Kiem`, `Muc_Do_Khieu_Nai_Ngam`, `Model_Su_Dung`.

### Task 2: `review_exact_match`, dọn `low_conf_threshold` ở NB06 & kết luận phân tích lỗi NB04
- [ ] Thêm `review_exact_match(y_true, y_pred)` vào `src/labels.py` (không tạo file mới).
- [ ] Trong `notebooks/04_phobert_multitask.ipynb`:
  - Thêm cell markdown `### Kết luận phân tích lỗi` ngay sau phần câu sai `PRICE_SER Negative` (tóm tắt 4 ý: `PRICE_SER` Negative hay bị sót trong review dài/khen nhiều, `DESIGN_GEN` `None` dễ lẫn Neg/Pos, `Neutral` yếu do ít mẫu 78–147 trừ `PRICE_SER`, và `review_exact_match = 0.5567`).
  - Không train lại trên Colab.
- [ ] Trong `notebooks/06_excel_auto_detector.ipynb`: đổi khóa `low_conf_threshold` sang `review_conf_threshold` cho thống nhất, chạy lại NB06.

### Task 3: Lọc từ khóa cụm NB05 không trùng & bỏ mảnh từ
- [ ] Trong `notebooks/05_shap_and_actionable_insights.ipynb` (cell 12):
  - Thêm mảnh từ `{'bảo', 'triệu', 'nghìn', 'đồng'}` vào `VN_STOP`.
  - Chấm điểm log-odds theo cụm rồi gán mỗi từ cho **duy nhất 1 cụm** có điểm cao nhất trước khi lấy `top_k=6`.
- [ ] Chạy lại NB05, kiểm tra `data/processed/actionable_insights.csv` không còn từ lặp giữa 5 cụm.
- [ ] Thêm test kiểm tra không trùng từ khóa & không có mảnh từ vào `tests/test_v4_release_checks.py`.

### Task 4: Đồng bộ số liệu `README.md` & `TAI_LIEU_BAN_GIAO_APP.md`
- [ ] Cập nhật `README.md` khớp hoàn toàn với `models/model_metadata.json`:
  - `absa_macro_f1 = 0.8410`, `mean_4cls_macro_f1 = 0.7924`, `review_exact_match = 0.5567` (`1 - 0.4433`), `needs_review` (`15.7%` gắn cờ, `74.5%` precision, `26.4%` error recall).
  - Bổ sung cảnh báo: `dev_pred_phobert.csv` chứa các dòng đã dùng cho early stopping/calibration, không dùng để báo cáo kết quả.
  - Bổ sung mục `## Tải weights` (chừa chỗ điền link Drive `WEIGHTS_URL` nếu cần).
- [ ] Thêm test kiểm tra số liệu `README.md` khớp `model_metadata.json` vào `tests/test_v4_release_checks.py`.

### Task 5: Kiểm thử & Push
- [ ] Chạy toàn bộ `pytest tests/ -v` (31 test cũ + toàn bộ test mới đều PASS).
- [ ] Commit & `git push origin main`.
