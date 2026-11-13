# Kế Hoạch Triển Khai Hệ Thống Phân Tích Đánh Giá Khách Hàng & Đề Xuất Cải Thiện Sản Phẩm TMĐT (VoC-Insight)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hoàn chỉnh hệ thống Khoa học Dữ liệu & AI phân tích đánh giá khách hàng TMĐT dựa trên bộ dữ liệu chuẩn UIT-ViSFD (11,122 đánh giá), bao gồm bộ 7 Notebooks (.ipynb) chuẩn phong cách coding_style.md, mô hình học đa nhiệm PhoBERT ABSA, giải thích SHAP XAI, tự động nhận diện 10 file Excel bẩn và tài liệu bàn giao Web App 2 vai trò.

**Architecture:** Pipeline 7 Jupyter Notebooks độc lập trao đổi qua file CSV trong `data/raw` và `data/processed`, mô hình chính `PhoBERT Multi-Task` (dự đoán Cảm xúc 3 lớp + Khía cạnh 5 nhóm), bộ mô hình cơ sở TF-IDF + LinearSVC/LogisticRegression có tinh chỉnh tham số, động cơ gom cụm đề xuất 6 nhóm hành động thực tế, và thuật toán heuristics tự động giải mã file khảo sát bẩn.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers (`vinai/phobert-base-v2`), scikit-learn, SHAP, pandas, numpy, openpyxl, xlrd, xlwt, lxml, matplotlib, seaborn.

**Spec:** [Spec.md](file:///d:/E-commerce/Spec.md)

---

## Global Constraints

- Tuân thủ 100% quy ước kích hoạt trong [SKILL.md](file:///d:/E-commerce/SKILL.md).
- Tuân thủ nghiêm ngặt [coding_style.md](file:///d:/What-If%20Simulator/coding_style.md): cell ngắn 1–5 dòng, có cell `.info()`, `.head()`, `.value_counts()` ngay sau biến đổi, thử biến `tempt` trước khi `.apply()`, comment ngắn `#` không dấu cách, tiêu đề Markdown HOA, `random_state=42`.
- Chống rò rỉ dữ liệu (Zero Data Leakage): `TfidfVectorizer` và `class_weight` chỉ fit trên tập Train (`7,786` dòng), tập Dev (`1,112` dòng) dùng chọn mô hình/hyperparameter, tập Test (`2,224` dòng) chỉ mở ra đánh giá 1 lần ở cell cuối.
- Tất cả các notebook phải lưu sẵn kết quả chạy (executed outputs) để mở trên VS Code thấy ngay bảng biểu và đồ thị.

## Review Focus

- Tính toán nhãn đa khía cạnh: Đảm bảo trích xuất đúng nhãn cảm xúc và khía cạnh chính từ chuỗi nhãn gốc `{CAMERA#Positive};{BATTERY#Negative}` mà không làm mất dữ liệu.
- Xử lý mất cân bằng nhãn: Lớp thiểu số (Neutral và các khía cạnh ít xuất hiện) phải được bảo vệ bằng Smoothed Class Weights.
- Khả năng chạy được trên CPU cá nhân: Các hàm đọc Excel và mô hình TF-IDF Baseline + SHAP phải phản hồi dưới 1 giây.
- Khả năng tương thích Google Colab GPU: Notebook 04 phải có sẵn cờ `IN_COLAB` tự động tải dữ liệu và đóng gói `outputs.zip` tải về máy.
- Tự động nhận diện 10 file bẩn: Đạt tỷ lệ 10 / 10 file nhận diện đúng Sheet, dòng tiêu đề và cột nội dung.

---

## Danh Sách Các Nhiệm Vụ (Task Checklist)

### Task 1: Khởi tạo Cấu trúc Dự án & Dependencies
**Files:**
- Create: `d:\E-commerce\requirements.txt`
- Create directories: `data/raw`, `data/processed`, `data/samples_messy`, `models`, `notebooks`, `reports/figures`

**Steps:**
- [x] Tạo file `requirements.txt` chứa đầy đủ các thư viện (`torch`, `transformers`, `scikit-learn`, `shap`, `pandas`, `openpyxl`, `xlrd`, `xlwt`, `lxml`, `matplotlib`, `seaborn`, `nbclient`, `nbformat`).
- [x] Tạo đầy đủ cây thư mục dự án trong `d:\E-commerce\`.
- [x] Kiểm tra môi trường Python sẵn sàng cho các thư viện cốt lõi.

---

### Task 2: Notebook 00 — Tải & Chuẩn Hóa Dữ Liệu Gốc UIT-ViSFD (`00_load_and_merge_raw.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\00_load_and_merge_raw.ipynb`
- Output: `d:\E-commerce\data\raw\Train.csv`, `Dev.csv`, `Test.csv`, `d:\E-commerce\data\processed\{train,dev,test}_raw.csv`

**Steps:**
- [x] Viết script tải trực tiếp `UIT-ViSFD.zip` từ kho chính thức của tác giả trường UIT về `data/raw/` (hoặc từ URL backup).
- [x] Đọc và giải mã cột `label` gốc `{ASPECT#Polarity}`:
  - Tách thành nhãn Cảm xúc chính (`sentiment`: 0=Negative, 1=Neutral, 2=Positive).
  - Tách thành Khía cạnh chính (`aspect_group`: 0..4 theo 5 nhóm chuẩn).
- [x] Giữ nguyên điểm sao `n_star` (1–5 sao) và mốc thời gian `date_time`.
- [x] Xuất ra `data/processed/train_raw.csv` (7,786 dòng), `dev_raw.csv` (1,112 dòng), `test_raw.csv` (2,224 dòng).
- [x] Chạy thực thi notebook và xác minh các cell output `.info()`, `.head()`, `.value_counts()`.

---

### Task 3: Notebook 01 — Tiền Xử Lý Ngôn Ngữ TMĐT & Bảo Mật PII (`01_text_cleaning_and_pii.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\01_text_cleaning_and_pii.ipynb`
- Output: `d:\E-commerce\data\processed\{train,dev,test}_clean.csv`

**Steps:**
- [x] Chuẩn hóa Unicode `NFC` và chữ thường.
- [x] Thu gọn ký tự lặp kéo dài (`đẹpppp` $\rightarrow$ `đẹp`, `quáaaa` $\rightarrow$ `quá`).
- [x] Dịch từ điển Teencode TMĐT (`sp` $\rightarrow$ `sản phẩm`, `đt` $\rightarrow$ `điện thoại`, `auth` $\rightarrow$ `chính hãng`, `fake` $\rightarrow$ `hàng giả`, `ship` $\rightarrow$ `giao hàng`, `ko` $\rightarrow$ `không`, `dc` $\rightarrow$ `được`...).
- [x] Ẩn danh hóa PII: Che số điện thoại giao hàng `[PHONE]`, mã đơn hàng `[ORDER_ID]`, email `[EMAIL]`, tên gian hàng `[SHOP_NAME]`.
- [x] Đánh dấu câu vô nghĩa / spam xin xu (`is_no_content`).
- [x] Ghép từ tiếng Việt chuyên ngành công nghệ / TMĐT (`sentence_seg`) bằng dấu `_` (ví dụ `sản_phẩm`, `điện_thoại`, `tai_nghe`, `chính_hãng`, `hàng_giả`, `giao_hàng`, `thời_lượng_pin`, `màn_hình`, `chụp_ảnh`).
- [x] Xuất ra `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`.
- [x] Chạy thực thi và kiểm tra các cell thử nghiệm với biến `tempt`.

---

### Task 4: Notebook 02 — Khám Phá Dữ Liệu Đánh Giá TMĐT (`02_eda.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\02_eda.ipynb`
- Output: Biểu đồ trong `d:\E-commerce\reports\figures\` (`eda_sentiment_dist.png`, `eda_aspect_dist.png`, `eda_star_vs_sentiment.png`, `eda_aspect_sentiment_heatmap.png`)

**Steps:**
- [x] Thống kê phân bố Cảm xúc (3 lớp) và Khía cạnh (5 nhóm).
- [x] Thống kê tương quan giữa Số sao người mua chấm (`n_star` 1–5) với Cảm xúc văn bản (phát hiện trước tỷ lệ mâu thuẫn sao).
- [x] Vẽ và lưu 4 biểu đồ phân tích chất lượng cao vào `reports/figures/`.
- [x] Chạy thực thi và xác nhận hình ảnh hiển thị mượt mà trong notebook.

---

### Task 5: Notebook 03 — Vector Hóa TF-IDF & Các Mô Hình Baseline (`03_baseline_tfidf_models.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\03_baseline_tfidf_models.ipynb`
- Output: `models/tfidf_vectorizer.pkl`, `models/svc_sentiment.pkl`, `models/svc_aspect.pkl`, `models/lr_sentiment.pkl`, `data/processed/test_pred_baseline.csv`

**Steps:**
- [x] Kiểm tra giao nhau giữa Train và Test để chứng minh không trùng lặp (Anti-leakage check).
- [x] Khởi tạo `TfidfVectorizer(ngram_range=(1,2), max_features=12000, min_df=3, sublinear_tf=True)` fit **CHỈ TRÊN TRAIN**.
- [x] Tính Smoothed Class Weights cho Cảm xúc và Khía cạnh trên Train set.
- [x] Huấn luyện & Tinh chỉnh tham số bằng `RandomizedSearchCV(cv=3, scoring='f1_macro')`:
  - `ComplementNB`
  - `LogisticRegression`
  - `LinearSVC` (Model Baseline mốc chuẩn tốt nhất)
- [x] Chọn tham số tối ưu dựa trên tập **Dev**, sau đó mới đánh giá niêm phong trên tập **Test** (`2,224` dòng).
- [x] Xuất các file mô hình `.pkl` và bảng kết quả `test_pred_baseline.csv`.
- [x] Chạy thực thi notebook và in báo cáo `classification_report`.

---

### Task 6: Notebook 04 — Huấn Luyện PhoBERT Multi-Task ABSA (`04_phobert_multitask.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\04_phobert_multitask.ipynb`
- Output: `models/phobert_multitask/` (`phobert_multitask.pt`, tokenizer files), `models/model_metadata.json`, `data/processed/test_pred_phobert.csv`

**Steps:**
- [x] Thiết lập kiến trúc mạng nơ-ron `PhoBertMultiTask`:
  - Shared Encoder: `vinai/phobert-base-v2`.
  - Head 1: Phân loại Cảm xúc (3 classes).
  - Head 2: Phân loại Khía cạnh chính (5 classes).
- [x] Áp dụng `CrossEntropyLoss(weight=smoothed_weights, label_smoothing=0.05)`.
- [x] Thiết kế logic dò `best_hp` (learning rate, dropout, weight decay) trên tập Dev.
- [x] Huấn luyện chính thức với Linear Warmup + Cosine Scheduler và Early Stopping theo `dev_mean_f1`.
- [x] Cấu hình chạy được trên cả GPU (Colab/máy cá nhân) và CPU (inference).
- [x] Dự đoán tập Test, gắn cờ mâu thuẫn sao / độ tự tin thấp (`needs_review`).
- [x] Xuất file trọng số `phobert_multitask.pt`, metadata chỉ số so sánh và `test_pred_phobert.csv`.

---

### Task 7: Notebook 05 — Giải Thích SHAP XAI & Đề Xuất Cải Thiện Sản Phẩm (`05_shap_and_actionable_insights.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\05_shap_and_actionable_insights.ipynb`
- Output: `reports/figures/shap_top_negative_words.png`, `data/processed/actionable_insights.csv`

**Steps:**
- [x] Khởi tạo `shap.LinearExplainer` sử dụng tập Train làm background (tránh lỗi leakage).
- [x] Trích xuất Top 15 từ khóa gây ra đánh giá Tiêu cực mạnh nhất toàn bộ tập dữ liệu TMĐT (`reports/figures/shap_top_negative_words.png`).
- [x] Viết hàm giải thích cục bộ từng câu (tô màu từ ngữ Đỏ/Xanh) phản hồi dưới 20ms.
- [x] Xây dựng Động cơ gom cụm phản hồi tiêu cực thành **6 nhóm giải pháp hành động thực tế**:
  1. Tối ưu pin & nguồn sạc
  2. Kiểm soát chất lượng màn hình & camera
  3. Tinh chỉnh hiệu năng & nhiệt độ
  4. Nâng cấp quy chuẩn đóng gói & vận chuyển
  5. Điều chỉnh chính sách bảo hành, đổi trả & voucher đền bù
  6. Đồng bộ thông tin mô tả sản phẩm
- [x] Xuất file `data/processed/actionable_insights.csv`.
- [x] Chạy thực thi notebook và kiểm tra outputs.

---

### Task 8: Notebook 06 — Tự Động Nhận Diện 10 File Xuất Thô TMĐT Bẩn (`06_excel_auto_detector.ipynb`)
**Files:**
- Create & Execute: `d:\E-commerce\notebooks\06_excel_auto_detector.ipynb`
- Output: 10 file trong `data/samples_messy/`, `data/processed/sample_excel_predicted.xlsx`

**Steps:**
- [x] Sinh 10 file khảo sát TMĐT "bẩn" mô phỏng thực tế tại `data/samples_messy/`:
  - `01_shopee_clean.xlsx`
  - `02_tiki_semicolon_bom.csv`
  - `03_lazada_english.xlsx`
  - `04_tiktok_title_offset.xlsx`
  - `05_pos_legacy.xls`
  - `06_multi_sheet_report.xlsx`
  - `07_utf16_tab_export.csv`
  - `08_two_text_cols.xlsx`
  - `09_merged_cells_pagination.xlsx`
  - `10_html_table_as_xls.xls`
- [x] Xây dựng bộ heuristics tự động nhận diện:
  - `read_any_file`: Tự nhận diện encoding, delimiter và bảng HTML đội lốt `.xls`.
  - `pick_best_sheet`: Tự chọn sheet chứa dữ liệu đánh giá thực.
  - `detect_header_row`: Tự tìm dòng tiêu đề thật (bỏ qua dòng tiêu đề báo cáo sàn ở đầu).
  - `detect_text_columns`: Tự chấm điểm cột văn bản đánh giá và loại bỏ cột PII.
  - `detect_star_column`: Tự nhận diện cột số sao đánh giá (1–5 sao).
- [x] Chạy kiểm thử tự động trên 10 file: **Yêu cầu đạt pass rate 10 / 10 file (100%)**.
- [x] Tích hợp pipeline ẩn danh PII, gán cờ `needs_review` và xuất file mẫu `sample_excel_predicted.xlsx`.

---

### Task 9: Tài Liệu Bàn Giao Xây Dựng Web App 2 Vai Trò (`TAI_LIEU_BAN_GIAO_APP.md`)
**Files:**
- Create: `d:\E-commerce\TAI_LIEU_BAN_GIAO_APP.md`

**Steps:**
- [x] Soạn thảo chi tiết tài liệu bàn giao kỹ thuật cho nhóm Frontend/Backend:
  - Phân vai mô hình (PhoBERT Multi-Task chính vs Baseline TF-IDF siêu tốc).
  - Bảng quy đổi nhãn 5 khía cạnh + 3 cảm xúc.
  - 5 Luồng nghiệp vụ chi tiết (Tiền xử lý PII, Đánh giá thời gian thực, Tự động đọc Excel bẩn, Gắn cờ mâu thuẫn sao, Gợi ý hành động).
  - Thiết kế màn hình: Cổng Người mua hàng (3 nút câu mẫu để demo) + Dashboard Quản trị Doanh nghiệp (4 Tab chuyên sâu).
  - Cấu trúc REST API JSON chuẩn cho tất cả các endpoint (`/predict`, `/upload/inspect`, `/upload/analyze`, `/dashboard/initial`).
  - Danh sách 6 bài test nghiệm thu bàn giao.
