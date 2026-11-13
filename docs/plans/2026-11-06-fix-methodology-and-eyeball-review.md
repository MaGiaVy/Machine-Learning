# Kế Hoạch Tái Cấu Trúc Phương Pháp Luận & Phân Tích Lỗi Eyeball/Blackbox (VoC-Insight v2.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triệt tiêu hoàn toàn 5 lỗ hổng phương pháp luận (ép nhãn lệch phân phối, tách từ giả lập, bất nhất log PhoBERT, đảo dấu SHAP, đứt gãy pipeline dự đoán Excel) và tích hợp quy trình **Phân tích lỗi Eyeball vs. Blackbox Dev Set theo chuẩn Andrew Ng (*Machine Learning Yearning*)** vào hệ thống Notebook.

**Architecture:** Tái định nghĩa quy tắc gán nhãn đa khía cạnh dựa trên đa số phiếu và mật độ từ khóa tại Notebook 00; tích hợp bộ tách từ `pyvi` và mở rộng từ điển chuẩn hóa tại Notebook 01; tháo bỏ `sklearn.pipeline.Pipeline` trong Notebook 03 để tuân thủ nghiêm ngặt `coding_style.md` đồng thời bổ sung đánh giá Overfitting (`Train F1` vs `Dev F1`); chia tập `Dev` (1,112 mẫu) thành `Eyeball Dev Set` (350 mẫu) và `Blackbox Dev Set` (762 mẫu) để phân tích lỗi thủ công (Error Analysis table) mà không làm quá khớp tập kiểm định; đồng bộ thực thi thật cho Notebook 04, 05, 06.

**Tech Stack:** Python 3.10+, Pandas, NumPy, Scikit-learn, PyVi (`pyvi.ViTokenizer`), PyTorch, Transformers (`vinai/phobert-base-v2`), SHAP, OpenPyXL.

**Spec:** [Spec.md](file:///d:/E-commerce/Spec.md) & [coding_style.md](file:///d:/E-commerce/coding_style.md)

## Global Constraints

- **Tuân thủ tuyệt đối `coding_style.md`:**
  - Mỗi cell code chỉ từ **1 đến 8 dòng** (tuyệt đối không có cell > 10 dòng, trừ định nghĩa `nn.Module` của PyTorch).
  - Sau mỗi bước biến đổi phải có cell riêng kiểm tra `.info()`, `.head()`, `.value_counts()`, hoặc `.shape`.
  - Thử nghiệm trên biến tạm `tempt` trước khi `.apply()` lên toàn bộ cột.
  - Comment tiếng Việt ngắn gọn 1 dòng, bắt đầu bằng `#` dính liền chữ (`#Thử trên 1 câu mẫu`), chỉ đặt ở nơi có quyết định kỹ thuật.
  - Tiêu đề Markdown viết HOA ngắn gọn (`TÁCH TỪ TIẾNG VIỆT`, `PHÂN TÍCH LỖI EYEBALL DEV`), không dùng emoji hay đoạn văn dài.
  - Không dùng `sklearn.pipeline.Pipeline` hay hàm bọc đánh giá chung `def evaluate(...)`; viết tường minh từng bước.
  - Cố định `random_state=42` ở mọi phép chia và mô hình.
- **Nguyên tắc Eyeball / Blackbox Dev Set (Andrew Ng):**
  - Tập `Dev.csv` (1,112 mẫu) được chia cố định (`random_state=42`, stratified theo `sentiment`) thành:
    - **`df_eyeball` (350 mẫu ~ 31.5%):** Được phép in câu chữ (`sentence`), soi từng câu đoán sai (Error Analysis), thống kê các nguyên nhân lỗi (Sarcasm, Teencode sót, Nhãn gốc nhiễu, Câu đa ý trái chiều) để cải tiến feature/tiền xử lý.
    - **`df_blackbox` (762 mẫu ~ 68.5%):** Tuyệt đối **KHÔNG** in câu chữ ra xem bằng mắt; chỉ dùng để tính chỉ số định lượng (`Macro F1`, `Accuracy`) nhằm chọn mô hình và kiểm tra xem các chỉnh sửa từ `Eyeball Dev` có thực sự tổng quát hóa hay đang overfit vào tập Eyeball.

## Review Focus

- Đánh giá 4–5 sao có nhiều khía cạnh khen và chỉ 1 khía cạnh góp ý nhỏ không còn bị ép 100% về nhãn `sentiment = 0` (Tiêu cực).
- Khía cạnh chính (`aspect_group`) được chọn theo sự xuất hiện thực tế và mức độ nhấn mạnh trong câu thay vì lấy phần tử đứng đầu theo thứ tự cột cố định của UIT-ViSFD (`SCREEN -> CAMERA -> ...`).
- Văn bản đưa vào `TfidfVectorizer` và `PhoBERT` được tách từ chuẩn bằng `pyvi.ViTokenizer.tokenize` thay vì danh sách 27 từ cứng.
- Top từ khóa SHAP giải thích cho dự đoán Tiêu cực (`sentiment = 0`) chỉ lấy các đặc trưng có giá trị SHAP dương (`sv > 0`) đẩy xác suất về phía lớp Tiêu cực, không dùng `-abs(x[1])`.
- Toàn bộ 100% đánh giá tiêu cực trên tập Test được phân bổ đầy đủ vào 6 nhóm giải pháp trong `actionable_insights.csv` (không bỏ sót 42.3% dữ liệu và không gán nhầm câu chê trầy vỏ máy/nóng ốp lưng vào nhóm Vận chuyển/Quà tặng).
- Dữ liệu Excel thực tế trong Notebook 06 được gộp đa cột văn bản, chạy qua chuỗi làm sạch + che PII (`[PHONE]`, `[EMAIL]`, `[ORDER_ID]`) + tách từ trước khi đưa vào `tfidf.transform()`.

---

### Task 1: Thiết kế lại Logic Quy đổi Nhãn Đa khía cạnh tại Notebook 00

**Files:**
- Modify: `notebooks/00_load_and_merge_raw.ipynb`
- Output: `data/processed/train_raw.csv`, `data/processed/dev_raw.csv`, `data/processed/test_raw.csv`
- Test: `tests/test_task1.py`

**Interfaces:**
- Consumes: `data/raw/Train.csv`, `data/raw/Dev.csv`, `data/raw/Test.csv`
- Produces: Các file `*_raw.csv` với cột `sentiment` (`0`: Tiêu cực, `1`: Trung lập/Hỗn hợp, `2`: Tích cực) và `aspect_group` (`0..4`) phản ánh đúng bản chất câu đánh giá.

- [ ] **Step 1: Cập nhật bài test kiểm tra phân phối nhãn trong `tests/test_task1.py`**

```python
def test_label_parsing_no_extreme_negative_bias():
    df_tr = pd.read_csv('data/processed/train_raw.csv')
    # Đánh giá 5 sao không được có quá 8% bị gán nhãn Tiêu cực (trước đây là 20.9%)
    star5 = df_tr[df_tr['n_star'] == 5]
    neg_ratio_5star = (star5['sentiment'] == 0).mean()
    assert neg_ratio_5star < 0.08, f"5-star negative ratio too high: {neg_ratio_5star:.2%}"
    # Lớp Trung lập / Hỗn hợp (sentiment == 1) phải chiếm ít nhất 8% tập Train (trước đây bị ép còn 2.86%)
    neu_ratio = (df_tr['sentiment'] == 1).mean()
    assert neu_ratio >= 0.08, f"Neutral/Mixed class collapsed: {neu_ratio:.2%}"
```

- [ ] **Step 2: Chạy pytest để xác nhận bài test FAIL trên dữ liệu cũ**

Run: `pytest tests/test_task1.py -v`
Expected: FAIL với `AssertionError: 5-star negative ratio too high: 20.90%`

- [ ] **Step 3: Sửa hàm `parse_label(row)` và tách các cell dài trong `notebooks/00_load_and_merge_raw.ipynb`**
  - Cập nhật quy tắc xác định `sentiment`:
    - Đếm số lượng nhãn `neg_cnt = sents.count(0)`, `pos_cnt = sents.count(2)`, `neu_cnt = sents.count(1)`.
    - Nếu câu chỉ có 1 phe (`neg_cnt > 0 and pos_cnt == 0` $\rightarrow$ `0`; `pos_cnt > 0 and neg_cnt == 0` $\rightarrow$ `2`).
    - Nếu câu vừa khen vừa chê (`neg_cnt > 0 and pos_cnt > 0`):
      - Nếu `neg_cnt > pos_cnt` hoặc `n_star <= 2` $\rightarrow$ `0` (Tiêu cực chiếm ưu thế).
      - Nếu `pos_cnt >= 2 * neg_cnt` và `n_star >= 5` $\rightarrow$ `2` (Khen vượt trội 5 sao, chỉ góp ý phụ).
      - Các trường hợp vừa khen vừa chê còn lại (đặc biệt ở mức 3–4 sao) $\rightarrow$ `1` (`Trung lập / Hỗn hợp`).
  - Cập nhật quy tắc chọn `aspect_group` đại diện (chống thiên kiến thứ tự cột `SCREEN -> CAMERA` của UIT-ViSFD):
    - Lọc danh sách các khía cạnh mang cùng cực cảm xúc với `sentiment` vừa chọn (nếu `sentiment == 1` và có khía cạnh bị chê `0` thì ưu tiên lấy khía cạnh bị chê để doanh nghiệp biết điểm cần cải thiện).
    - Nếu có nhiều khía cạnh cùng thỏa mãn, chọn khía cạnh có từ khóa xuất hiện đầu tiên trong câu văn bản `comment` (dựa trên bảng từ khóa định vị nhanh cho 5 nhóm) thay vì lấy `index(0)` theo thứ tự schema.
  - Chia nhỏ mọi cell > 10 dòng và thực thi lại toàn bộ Notebook 00 để xuất `train_raw.csv`, `dev_raw.csv`, `test_raw.csv`.

- [ ] **Step 4: Chạy lại pytest để xác nhận PASS**

Run: `pytest tests/test_task1.py -v`
Expected: PASS

---

### Task 2: Tích hợp Tách từ `pyvi`, Mở rộng Teencode & Chuẩn hóa PII tại Notebook 01 + Cập nhật EDA tại Notebook 02

**Files:**
- Modify: `notebooks/01_text_cleaning_and_pii.ipynb`, `notebooks/02_eda.ipynb`
- Output: `data/processed/train_clean.csv`, `data/processed/dev_clean.csv`, `data/processed/test_clean.csv`, `reports/figures/*.png`
- Test: `tests/test_task1.py`

**Interfaces:**
- Consumes: `train_raw.csv`, `dev_raw.csv`, `test_raw.csv`
- Produces: Cột `sentence_clean` (đã dịch đầy đủ teencode, che PII `[PHONE]`, `[EMAIL]`, `[ORDER_ID]`) và `sentence_seg` (tách từ tiếng Việt bằng `pyvi.ViTokenizer.tokenize`).

- [ ] **Step 1: Bổ sung test kiểm tra tách từ `pyvi`, teencode `tgdđ` và regex nguyên âm `ư` trong `tests/test_task1.py`**

```python
def test_pyvi_segmentation_and_teencode():
    df_te = pd.read_csv('data/processed/test_clean.csv')
    # Không còn từ viết tắt 'tgdđ' sót lại trong sentence_clean
    has_tgdd = df_te['sentence_clean'].fillna('').str.contains(r'\btgdđ\b').any()
    assert not has_tgdd, "Teencode 'tgdđ' was not normalized"
    # Kiểm tra các từ ghép ngoài danh sách 27 từ cũ (vd: 'nhiệt_tình', 'trải_nghiệm', 'phần_mềm') được pyvi nối '_'
    seg_all = " ".join(df_te['sentence_seg'].fillna(''))
    assert "nhiệt_tình" in seg_all and "trải_nghiệm" in seg_all
```

- [ ] **Step 2: Chạy pytest để xác nhận FAIL**

Run: `pytest tests/test_task1.py::test_pyvi_segmentation_and_teencode -v`
Expected: FAIL

- [ ] **Step 3: Cập nhật `notebooks/01_text_cleaning_and_pii.ipynb` và `notebooks/02_eda.ipynb`**
  - Tại Notebook 01:
    - Bổ sung vào `teencode_dict`: `'tgdđ': 'thế giới di động'`, `'đk': 'được'`, `'dk': 'được'`, `'hk': 'không'`, `'kg': 'không'`, `'mn': 'mọi người'`, `'bth': 'bình thường'`, `'sd': 'sử dụng'`, `'trc': 'trước'`, `'ttbh': 'trung tâm bảo hành'`, `'nv': 'nhân viên'`, `'bh': 'bảo hành'`, `'dt': 'điện thoại'`.
    - Cập nhật regex kéo dài âm tiết: thêm nhóm nguyên âm `ư` (`r'([ùúủũụưừứửữự])[uư]+'`), và rút gọn phụ âm lặp cuối từ (`r'([ptcmng])[ptcmng]+(?=\s|$)'`).
    - Mở rộng regex `[ORDER_ID]` để bắt cả mã đơn dạng `r'\b(dh|đh|ord|spx)[0-9a-z]{4,}\b'` và SĐT có dấu chấm/khoảng trắng `r'(0|\+84)[\s\.]?[0-9]{3}[\s\.]?[0-9]{3}[\s\.]?[0-9]{3,4}\b'`.
    - Thay thế vòng lặp 27 từ ghép trong `segment_vn` bằng `ViTokenizer.tokenize(str(t))` của thư viện `pyvi`.
    - Đảm bảo mọi cell đều ngắn $\le 8$ dòng theo đúng `coding_style.md`.
  - Tại Notebook 02:
    - Chạy lại toàn bộ biểu đồ EDA trên `train_clean.csv` mới và bổ sung 1 cell tính phân vị 95% & 99% của `word_count` (`df_train['word_count'].quantile([0.95, 0.99])`) để làm cơ sở khoa học cho ngưỡng `MAX_LEN = 128` ở Notebook 04.

- [ ] **Step 4: Chạy lại pytest để xác nhận PASS**

Run: `pytest tests/test_task1.py -v`
Expected: PASS

---

### Task 3: Chống Overfitting, Tuân thủ `coding_style.md` & Phân tích Lỗi Eyeball vs. Blackbox Dev Set tại Notebook 03

**Files:**
- Modify: `notebooks/03_baseline_tfidf_models.ipynb`
- Output: `models/tfidf_vectorizer.pkl`, `models/lr_sentiment.pkl`, `models/lr_aspect.pkl`, `models/svc_sentiment.pkl`, `models/svc_aspect.pkl`, `data/processed/test_pred_baseline.csv`
- Test: `tests/test_task2.py`

**Interfaces:**
- Consumes: `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`
- Produces: Vectorizer, các mô hình tuyến tính, bảng phân tích lỗi Eyeball Dev vs. Blackbox Dev, và dự đoán `test_pred_baseline.csv`.

- [ ] **Step 1: Cập nhật `tests/test_task2.py` để kiểm tra sự tồn tại của chia tách Eyeball/Blackbox và kiểm soát Overfitting**

```python
def test_notebook03_eyeball_blackbox_and_train_eval():
    with open('notebooks/03_baseline_tfidf_models.ipynb', encoding='utf-8') as f:
        nb_text = f.read()
    assert 'df_eyeball' in nb_text and 'df_blackbox' in nb_text, "Missing Eyeball/Blackbox split in NB03"
    assert 'f1_train' in nb_text or 'tr_f1' in nb_text, "Missing Train F1 evaluation to check overfitting"
```

- [ ] **Step 2: Chạy pytest để xác nhận FAIL**

Run: `pytest tests/test_task2.py -v`
Expected: FAIL

- [ ] **Step 3: Tái cấu trúc `notebooks/03_baseline_tfidf_models.ipynb` theo đúng chuẩn `coding_style.md` và Andrew Ng Error Analysis**
  - **Loại bỏ `sklearn.pipeline.Pipeline`:** Để tuân thủ Mục 10 của `coding_style.md` (*"Tuyệt đối tránh dùng Pipeline bọc hết preprocessing -> Làm tay từng bước"*) mà vẫn không bị Data Leakage, thực hiện fit `tfidf` **chỉ trên `df_train['sentence_seg']`**, và đánh giá chọn siêu tham số trực tiếp trên tập Validation (`Dev`) thay vì 3-fold CV nội bộ.
  - **Chia tập `Dev` thành `Eyeball Dev` và `Blackbox Dev`:**
    ```python
    #Chia dev thanh eyeball (350 mau) de soi loi va blackbox (762 mau) de chot mo hinh
    df_eyeball, df_blackbox = train_test_split(df_dev, train_size=350, stratify=df_dev['sentiment'], random_state=42)
    ```
  - **Kiểm tra Overfitting đầy đủ cho từng mô hình (`ComplementNB`, `LogisticRegression`, `LinearSVC`):**
    - Truyền đúng `smooth_w_sent` và `smooth_w_asp` vào cả `LogisticRegression` và `LinearSVC`, giới hạn `C` trong `[0.1, 0.5, 1.0, 2.0]` để chống overfit.
    - Với mỗi mô hình, tính và `print` rõ ràng 3 con số: `Train F1`, `Eyeball Dev F1`, và `Blackbox Dev F1`.
  - **Thêm mục `PHÂN TÍCH LỖI TRÊN TẬP EYEBALL DEV (ANDREW NG)`:**
    - Lọc các mẫu trong `df_eyeball` bị đoán sai (`y_eye != pred_eye`).
    - In ra 10–15 câu sai điển hình kèm `n_star`, nhãn thật, nhãn đoán.
    - Phân loại các mẫu lỗi trong `df_eyeball` vào bảng thống kê tỷ lệ % các nhóm nguyên nhân (câu đa ý trái chiều, từ vựng hiếm/ẩn dụ, teencode/viết tắt, ngắn thiếu ngữ cảnh) để giải thích hướng cải tiến lên PhoBERT.
  - Chọn mô hình có `Blackbox Dev F1` cao nhất để dự đoán trên `X_te_tfidf` và lưu `test_pred_baseline.csv`.

- [ ] **Step 4: Chạy lại pytest để xác nhận PASS**

Run: `pytest tests/test_task2.py -v`
Expected: PASS

---

### Task 4: Đồng bộ Thực thi Thật, Xóa Log Giả lập & So sánh Eyeball/Blackbox tại Notebook 04

**Files:**
- Modify: `notebooks/04_phobert_multitask.ipynb`
- Output: `data/processed/test_pred_phobert.csv`, `models/model_metadata.json`
- Test: `tests/test_task3.py`

**Interfaces:**
- Consumes: `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`, `models/phobert_multitask/phobert_multitask.pt`
- Produces: Kết quả đánh giá thực tế trên `Eyeball Dev`, `Blackbox Dev`, `Test` đồng bộ 100% giữa Notebook output, `test_pred_phobert.csv` và `model_metadata.json`.

- [ ] **Step 1: Cập nhật `tests/test_task3.py` kiểm tra tính đồng bộ của `04_phobert_multitask.ipynb`**

```python
def test_nb04_executed_and_no_fake_logs():
    import json
    with open('notebooks/04_phobert_multitask.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    # Mọi cell có code phải có execution_count != None
    assert all(c.get('execution_count') is not None for c in code_cells), "NB04 has unexecuted cells"
    # Không còn biến training_logs gán cứng
    full_src = "\n".join("".join(c['source']) for c in code_cells)
    assert "training_logs = [" not in full_src, "Hardcoded training_logs still present in NB04"
```

- [ ] **Step 2: Chạy pytest để xác nhận FAIL**

Run: `pytest tests/test_task3.py -v`
Expected: FAIL

- [ ] **Step 3: Sửa và chạy thực thi toàn bộ `notebooks/04_phobert_multitask.ipynb`**
  - Tại Cell 5 (nhánh fallback Colab): gọi `ViTokenizer.tokenize` cho `sentence_seg` để đồng nhất 100% với Notebook 01.
  - Tại Cell 12: xóa bỏ đoạn giả lập `hp_candidates` gán cứng `best_hp = hp_candidates[1]`; thay bằng thiết lập cấu hình chuẩn từ thực nghiệm (`lr=2e-5`, `dropout=0.3`, `weight_decay=0.02`).
  - Tại Cell 14: trong nhánh GPU, bổ sung code tính `Train Macro F1` thật và cơ chế `EarlyStopping` lưu lịch sử vào `models/phobert_multitask/training_history.json`; khi tải checkpoint `phobert_multitask.pt` trên máy local, đọc từ `training_history.json` và chạy đánh giá thật (inference thật bằng PyTorch hoặc nạp dự đoán đã xác thực) trên `df_eyeball`, `df_blackbox`, và `df_test`.
  - Bổ sung cell đối chiếu lỗi trên tập `df_eyeball`: chứng minh PhoBERT khắc phục được bao nhiêu % số câu lỗi ngữ cảnh / đảo ngữ mà TF-IDF đã bó tay ở Notebook 03, và kiểm chứng mức tăng F1 độc lập trên `df_blackbox`.
  - Tách các cell dài và chạy thực thi toàn bộ Notebook 04 bằng `jupyter nbconvert --to notebook --execute --inplace` để mọi cell có `execution_count` và `outputs` khớp tuyệt đối với `model_metadata.json` và `test_pred_phobert.csv`.

- [ ] **Step 4: Chạy lại pytest để xác nhận PASS**

Run: `pytest tests/test_task3.py -v`
Expected: PASS

---

### Task 5: Sửa lỗi Đảo dấu SHAP Local & Bao phủ 100% Đánh giá Tiêu cực trong Actionable Insights tại Notebook 05

**Files:**
- Modify: `notebooks/05_shap_and_actionable_insights.ipynb`
- Output: `data/processed/actionable_insights.csv`, `reports/figures/shap_top_negative_words.png`
- Test: `tests/test_task4.py`

**Interfaces:**
- Consumes: `data/processed/test_pred_phobert.csv`, `models/tfidf_vectorizer.pkl`, `models/lr_sentiment.pkl`
- Produces: Giải thích từ khóa tiêu cực chính xác (`sv > 0`) và bảng `actionable_insights.csv` bao phủ 100% đánh giá tiêu cực không gán nhầm ngữ nghĩa.

- [ ] **Step 1: Cập nhật `tests/test_task4.py` để kiểm tra dấu SHAP dương và độ bao phủ 100% của Actionable Insights**

```python
def test_actionable_insights_full_coverage_and_positive_shap():
    df_pred = pd.read_csv('data/processed/test_pred_phobert.csv')
    df_ins = pd.read_csv('data/processed/actionable_insights.csv')
    total_neg = int((df_pred['pred_sentiment'] == 0).sum())
    # Tổng số lượng câu tiêu cực trong 6 nhóm giải pháp phải bao phủ 100% tổng số câu tiêu cực
    assert df_ins['negative_count'].sum() == total_neg, (
        f"Insights coverage mismatch: {df_ins['negative_count'].sum()} != {total_neg}"
    )
    # Kiểm tra trong NB05 không dùng -abs(x[1]) khi lấy từ khóa đóng góp vào lớp Tiêu cực
    with open('notebooks/05_shap_and_actionable_insights.ipynb', encoding='utf-8') as f:
        nb_src = f.read()
    assert '-abs(x[1])' not in nb_src, "Local SHAP explanation still uses -abs(x[1])"
```

- [ ] **Step 2: Chạy pytest để xác nhận FAIL**

Run: `pytest tests/test_task4.py -v`
Expected: FAIL

- [ ] **Step 3: Sửa logic SHAP và thiết kế lại 6 cụm Actionable Insights trong `notebooks/05_shap_and_actionable_insights.ipynb`**
  - **Sửa lỗi SHAP Local (Cell 10–11):**
    - Chọn mẫu mà **cả PhoBERT và mô hình tuyến tính đều dự đoán là Tiêu cực (`0`)** để đảm bảo tính nhất quán khi giải thích bằng `LinearExplainer` (Global/Local Lexical Explainer).
    - Chỉ lọc các đặc trưng có `sv_neg_one[i] > 0` (đẩy dự đoán về phía Tiêu cực) và sắp xếp theo `key=lambda x: -x[1]`.
  - **Chuẩn hóa 6 nhóm Giải pháp Hành động (Cells 12–15) bám sát thực tế dữ liệu và bao phủ 100% `df_neg`:**
    - Sử dụng phân vùng độc quyền (mutually exclusive & collectively exhaustive partitioning) trên tập `df_neg`:
      1. `asp_id == 0` (`Hiệu năng & Cấu hình`): Tinh chỉnh hiệu năng, xử lý giật lag & tối ưu đa nhiệm.
      2. `asp_id == 2` (`Thời lượng Pin & Sạc`): Tối ưu tiêu thụ pin, kiểm soát nhiệt độ & bộ sạc.
      3. `asp_id == 1` + khớp từ khóa Camera (`chụp|cam|ảnh|quay|mờ|nét|zoom|selfie` với ranh giới từ `(?<!\w)`): Nâng cấp thuật toán xử lý ảnh & chất lượng cụm Camera.
      4. `asp_id == 1` + các câu còn lại của nhóm 1 (`Màn hình & Hiển thị`): Kiểm soát chất lượng tấm nền màn hình, độ nhạy cảm ứng & hiển thị.
      5. `asp_id == 3` (`Giá cả, Bảo hành & Dịch vụ Hậu mãi`): Chuẩn hóa quy trình CSKH, bảo hành đổi trả & chính sách giá/khuyến mãi.
      6. `asp_id == 4` (`Thiết kế Ngoại quan & Hoàn thiện Sản phẩm`): Nâng cấp độ bền vật liệu khung viền, mặt lưng, phụ kiện đi kèm & kiểm수 ngoại quan trước khi giao.
    - Nhờ cơ chế phân hoạch trọn vẹn này, **100% đánh giá tiêu cực** đều được gom vào đúng nhóm giải pháp chuyên môn, không còn bất kỳ câu nào bị rơi rụng và không bao giờ gán nhầm câu chê "vỏ máy trầy xước" sang lỗi đơn vị vận chuyển.
  - Chia nhỏ các cell dài và chạy thực thi lại toàn bộ Notebook 05.

- [ ] **Step 4: Chạy lại pytest để xác nhận PASS**

Run: `pytest tests/test_task4.py -v`
Expected: PASS

---

### Task 6: Khắc phục Đứt gãy Tiền xử lý, Ẩn danh PII & Xử lý Merged Cells/Multi-Column tại Notebook 06

**Files:**
- Modify: `notebooks/06_excel_auto_detector.ipynb`
- Output: `data/processed/sample_excel_predicted.xlsx`
- Test: `tests/verify_fixes.py`

**Interfaces:**
- Consumes: `data/samples_messy/*`, `models/tfidf_vectorizer.pkl`, `models/svc_sentiment.pkl`, `models/svc_aspect.pkl`
- Produces: Bảng dữ liệu đã chuẩn hóa từ mọi định dạng Excel/CSV bẩn và file `sample_excel_predicted.xlsx` đã được che PII + tách từ chuẩn xác trước khi dự đoán.

- [ ] **Step 1: Bổ sung bài test kiểm tra PII Masking và Preprocessing trong `tests/verify_fixes.py`**

```python
def test_notebook06_pii_masked_and_preprocessed():
    df_out = pd.read_excel('data/processed/sample_excel_predicted.xlsx')
    assert 'sentence_clean' in df_out.columns, "NB06 output missing cleaned & PII-masked text column"
    # Đảm bảo không còn SĐT 10 số lộ thiên trong cột sentence_clean
    raw_phones = df_out['sentence_clean'].fillna('').str.contains(r'\b0[0-9]{9}\b').any()
    assert not raw_phones, "Unmasked phone numbers leaked into sample_excel_predicted.xlsx"
```

- [ ] **Step 2: Chạy pytest để xác nhận FAIL**

Run: `pytest tests/verify_fixes.py -v`
Expected: FAIL

- [ ] **Step 3: Cập nhật `notebooks/06_excel_auto_detector.ipynb` và thực thi lại toàn bộ**
  - Trong `build_clean_table`:
    - Lọc bỏ các dòng tiêu đề lặp lại ở giữa bảng (khi $\ge 50\%$ giá trị trong dòng trùng khớp với tên các cột `df_clean.columns` — xử lý triệt để file `09_merged_cells_pagination.xlsx`).
    - Thực hiện `.ffill()` trên cột phân loại/sản phẩm bị khuyết do gộp ô (merged cells).
    - Nếu `detect_text_columns` phát hiện $\ge 2$ cột văn bản (như file `08_two_text_cols.xlsx` có `Ưu điểm` và `Nhược điểm`), tự động nối các cột văn bản đó thành một cột thống nhất `combined_review` trước khi đưa vào pipeline.
  - Trước khi gọi `tfidf.transform()`:
    - Áp dụng đầy đủ hàm `clean_pipeline` (chuẩn hóa NFC, dịch teencode, **ẩn danh PII** thành `[PHONE]`, `[EMAIL]`, `[ORDER_ID]`) để tạo cột `sentence_clean`.
    - Áp dụng `ViTokenizer.tokenize` để tạo cột `sentence_seg` rồi mới truyền `df_sample['sentence_seg']` vào `tfidf.transform()`.
  - Tách các cell dài thành từng bước nhỏ theo đúng `coding_style.md` và chạy thực thi lại toàn bộ Notebook 06.

- [ ] **Step 4: Chạy toàn bộ bộ test nghiệm thu (`test_task1.py` đến `verify_fixes.py`)**

Run: `pytest tests/ -v`
Expected: PASS 100%
