# Kế Hoạch Triển Khai Chỉnh Sửa & Tối Ưu Toàn Diện Mã Nguồn VoC-Insight

> **Dành cho Agentic Workers:** SUB-SKILL BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyên dùng) hoặc `superpowers:executing-plans` để thực thi kế hoạch này theo từng task. Các bước theo dõi bằng cú pháp checkbox (`- [ ]`).

**Goal:** Khắc phục triệt để các lỗi logic, nguy cơ rò rỉ dữ liệu trong Cross-Validation, loại bỏ bắt nhầm đánh giá tích cực là spam và cân bằng hàm loss đa nhiệm cho toàn bộ pipeline của hệ thống VoC-Insight.

**Architecture:** Điều chỉnh logic tiền xử lý chuỗi văn bản trong Notebook 01; bọc Scikit-learn Pipeline trong Notebook 03 chống rò rỉ CV; cân bằng hệ số hàm mất mát Multi-Task và loại bỏ rủi ro fallback nhãn thật trong Notebook 04; chuẩn hóa bộ lọc Actionable Insights trong Notebook 05.

**Tech Stack:** Python 3.10+, Pandas, Scikit-learn, PyTorch, Transformers (PhoBERT), Regular Expressions, Jupyter Notebooks.

**Spec:** [Spec.md](file:///d:/E-commerce/Spec.md) & [Báo cáo Code Review](file:///d:/E-commerce/docs/plans/2026-11-01-ecommerce-feedback-analyzer.md)

## Global Constraints

- Môi trường thực thi tương thích hoàn toàn cả trên Google Colab (GPU T4) và Local VS Code.
- Giữ nguyên cấu trúc phân chia chuẩn UIT-ViSFD (7,786 train / 1,112 dev / 2,224 test).
- Tuyệt đối tuân thủ `coding_style.md`: code ngắn gọn, biến đổi có kiểm tra `info()`, `head()`, `value_counts()`, giữ `random_state=42`.

## Review Focus

- Không còn câu chứa từ "tốt", "tuyệt vời", "ok" bị đánh dấu `is_no_content = True`.
- Regex rút gọn ký tự kéo dài phải thu về 1 ký tự (`quáaaa` $\rightarrow$ `quá`) để tránh phát sinh từ OOV.
- TfidfVectorizer trong tinh chỉnh tham số Baseline không được rò rỉ dữ liệu fold kiểm định.
- Nhánh chạy trên CPU không được phép lấy nhãn ground-truth `df_test['sentiment']` để giả lập kết quả.
- Tỷ lệ gắn cờ `needs_review` trên tập test phản ánh chính xác các mâu thuẫn sao thực sự, giảm thiểu False Positive.

---

### Task 1: Sửa logic Tiền xử lý văn bản & Bộ lọc câu vô nghĩa (Notebook 01)

**Files:**
- Modify: `notebooks/01_text_cleaning_and_pii.ipynb`
- Output: `data/processed/train_clean.csv`, `data/processed/dev_clean.csv`, `data/processed/test_clean.csv`

**Interfaces:**
- Consumes: `data/processed/train_raw.csv`, `dev_raw.csv`, `test_raw.csv`
- Produces: `sentence_clean`, `sentence_seg`, `is_no_content` chuẩn hóa

- [x] **Step 1: Viết test kiểm tra logic regex rút gọn ký tự và lọc spam**
  - Kiểm tra `clean_text("máy dùng quáaaa đãaa")` phải ra `"máy dùng quá đã"`.
  - Kiểm tra `check_no_content("tốt")` và `check_no_content("tuyệt vời")` phải trả về `False`.
  - Kiểm tra `check_no_content("nhận xu")` và `check_no_content("cho 5 sao nhận xu")` phải trả về `True`.

- [x] **Step 2: Cập nhật Cell 10 & 15 trong Notebook 01**
  - Đổi `re.sub(r'([a-zÀ-ỹ])\1{2,}', r'\1\1', t)` thành `re.sub(r'([a-zÀ-ỹ])\1{2,}', r'\1', t)`.

- [x] **Step 3: Cập nhật danh sách `spam_patterns` tại Cell 16 trong Notebook 01**
  - Xóa bỏ các từ đánh giá thật `'ok'`, `'tốt'`, `'tuyệt vời'` ra khỏi danh sách `spam_patterns`.
  - Chỉ giữ: `['nhận xu', 'chưa dùng', 'cho 5 sao nhận xu', 'đã nhận hàng', 'không có ý kiến', 'chưa thử']`.

- [x] **Step 4: Chạy lại toàn bộ Notebook 01 và ghi đè các file `clean.csv`**
  - Thực thi Notebook 01 để xuất lại `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`.
  - Xác nhận lại số lượng `is_no_content` không còn tính nhầm các câu khen tích cực ngắn.

---

### Task 2: Chống rò rỉ dữ liệu trong Cross-Validation Baseline (Notebook 03)

**Files:**
- Modify: `notebooks/03_baseline_tfidf_models.ipynb`
- Output: `models/tfidf_vectorizer.pkl`, `models/svc_sentiment.pkl`, `models/svc_aspect.pkl`, `data/processed/test_pred_baseline.csv`

**Interfaces:**
- Consumes: `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`
- Produces: Mô hình Baseline chuẩn mực không rò rỉ CV

- [x] **Step 1: Viết script kiểm tra Pipeline Scikit-learn**
  - Đảm bảo `RandomizedSearchCV` nhận `Pipeline([('tfidf', vec), ('clf', model)])` và truyền dữ liệu văn bản thô `df_train['sentence_seg']` thay vì ma trận `X_tr_tfidf` đã fit toàn bộ.

- [x] **Step 2: Cập nhật Cell 18, 19, 21, 22 trong Notebook 03**
  - Cấu hình tham số dạng `clf__C`, `clf__class_weight` cho Pipeline.
  - Chạy `RandomizedSearchCV` trên `df_train['sentence_seg'].fillna('')`.

- [x] **Step 3: Đánh giá trên tập Dev & Test và xuất kết quả**
  - Ghi nhận F1-score thực tế và xuất lại `test_pred_baseline.csv`.

---

### Task 3: Cân bằng hàm Loss Multi-Task & Xóa bỏ Fallback nhãn thật (Notebook 04)

**Files:**
- Modify: `notebooks/04_phobert_multitask.ipynb`
- Output: `data/processed/test_pred_phobert.csv`, `models/model_metadata.json`

**Interfaces:**
- Consumes: Checkpoint `models/phobert_multitask/phobert_multitask.pt` hoặc huấn luyện trên GPU
- Produces: `test_pred_phobert.csv` với cờ `needs_review` chính xác

- [x] **Step 1: Sửa trọng số hàm Loss tại Cell 20 trong Notebook 04**
  - Điều chỉnh hệ số loss Aspect: đổi từ `1.2 * loss_asp_fn` thành `0.9 * loss_asp_fn` (hoặc `1.0`) để phản ánh đúng tương quan 3 lớp Sentiment vs 5 lớp Aspect.

- [x] **Step 2: Sửa logic nhánh CPU Fallback tại Cell 22 trong Notebook 04**
  - Xóa dòng fallback tự gán nhãn ground-truth `df_test['sentiment'].tolist()`.
  - Nếu không tìm thấy file cache và không có GPU, ném cảnh báo rõ ràng yêu cầu tải checkpoint hoặc chạy trên Google Colab.

- [x] **Step 3: Cập nhật cờ `needs_review` tại Cell 26 và ghi đè `test_pred_phobert.csv`**
  - Sử dụng cột `is_no_content` mới đã sửa ở Task 1.
  - Tính toán lại tỷ lệ `needs_review` thực tế.

- [x] **Step 4: Cập nhật `model_metadata.json`**
  - Ghi lại các chỉ số đánh giá Macro F1 và Accuracy chính xác.

---

### Task 4: Chuẩn hóa Động cơ Đề xuất Hành động Actionable Insights (Notebook 05)

**Files:**
- Modify: `notebooks/05_shap_and_actionable_insights.ipynb`
- Output: `data/processed/actionable_insights.csv`, `reports/figures/shap_top_negative_words.png`

**Interfaces:**
- Consumes: `data/processed/test_pred_phobert.csv`, `models/tfidf_vectorizer.pkl`, `models/svc_sentiment.pkl`
- Produces: `actionable_insights.csv` không bị đếm trùng lặp

- [x] **Step 1: Sửa điều kiện định tuyến tại Cell 18 trong Notebook 05**
  - Đảm bảo các nhóm hành động thuộc `asp_id == 4` (Đóng gói & Vận chuyển, Đồng bộ mô tả) chỉ lọc trên tập con `sub = df_neg[df_neg['pred_aspect'] == 4]` thay vì quét toàn bộ `df_neg`.

- [x] **Step 2: Chạy lại Notebook 05 và xuất file `actionable_insights.csv`**
  - Kiểm tra bảng phân phối số lượng phản hồi tiêu cực của 6 cụm giải pháp.

---

### Task 5: Xác minh và Kiểm định Toàn diện (Verification)

**Files:**
- Create: `tests/verify_fixes.py`

- [x] **Step 1: Viết script tự động kiểm tra nghiệm thu toàn bộ 4 task**
  - Xác nhận 0 câu chứa "tốt" hoặc "tuyệt vời" bị gắn cờ `is_no_content`.
  - Xác nhận tỷ lệ `needs_review` trên tập test không bị phóng đại.
  - Xác nhận các file output `train_clean.csv`, `test_pred_phobert.csv`, `actionable_insights.csv`, `model_metadata.json` đều tồn tại và đồng bộ.

- [x] **Step 2: Chạy script nghiệm thu và xác nhận PASS toàn bộ**
