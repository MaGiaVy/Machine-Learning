# Thiết Kế Kiến Trúc & Phương Pháp Luận Chuẩn Data Science — VoC-Insight v3.1 (Final Spec)

**Ngày thiết kế:** 2026-11-11  
**Trạng thái:** Đã vá đủ 7 điểm kiểm định thực nghiệm — Chờ người dùng phê duyệt để lập Implementation Plan (`writing-plans`)  
**Mục tiêu:** Giải quyết triệt để 100% các lỗ hổng về Data Leakage, Overfitting, Lệch nhãn, Lỗi Regex NLP, Lỗi toán học SHAP, Bất đồng bộ phiên bản (`Split-Brain`) và Lỗi Ingestion trên Notebook 00–06, thiết lập package dùng chung `src/` và huấn luyện lại **PhoBERT Multi-Task (Pure Sentiment + Multi-Label Aspect)** một lần duy nhất trên Google Colab GPU.

---

## 1. Mục Tiêu, Phạm Vi & Nguyên Tắc Thiết Kế (Goals & Non-Goals)

### Goals (Mục tiêu bắt buộc)
1. **Zero Target Leakage:** Loại bỏ hoàn toàn `n_star` khỏi quy trình sinh nhãn `sentiment` tại Notebook 00. Áp dụng công thức **Majority-over-3 thuần từ nhãn `{ASPECT#Polarity}`** (hoặc thống kê rõ `~19.6%` Trung lập/Hỗn hợp với Majority-over-3, hoặc `~12.8%` nếu giữ ưu tiên tuyệt đối không có cực đối lập — Spec này chọn **Majority-over-3** để đối xử công bằng cả 3 cực tính), khôi phục **~19.6% lớp Trung lập / Hỗn hợp** và **~144 ca mâu thuẫn sao thực sự** trên tập Test (`5.5%–6.5%`) để tính năng `needs_review` có ý nghĩa khoa học thực tế.
2. **Đúng bản chất ABSA Đa khía cạnh (85.3% câu $\ge 2$ khía cạnh):** Chuyển đầu ra Khía cạnh sang **Multi-Label Classification (5 nút nhị phân độc lập với `BCEWithLogitsLoss`)**, giữ `primary_aspect` làm trường dẫn xuất tương thích UI (kèm fallback `argmax` nếu cả 5 đầu đều `< threshold`), và báo cáo song song `Macro F1 (Multi-Label)` + `Hit-Any-True-Group`.
3. **Single Source of Truth (`src/` Package):** Đóng gói toàn bộ logic dùng chung thành package `src/` gồm `src/__init__.py`, `src/labels.py`, `src/text_cleaner.py`, `src/excel_ingestion.py`, `src/modeling.py`. Cả Local Notebooks, Colab Notebook 04 (clone/upload cùng thư mục `src/` — **tuyệt đối không viết code backup inline**), Pytest, và Backend FastAPI đều import trực tiếp từ `src/`.
4. **Zero Validation Leakage & Tách bạch Dev Set:**
   * Chia `df_dev` (`1,112` mẫu) theo `stratify` dựa trên `sentiment` + `primary_aspect` (fallback nếu nhóm quá nhỏ) thành **3 tập con không chồng lấn**:
     * **`df_eyeball` (250 mẫu):** Soi lỗi thủ công bằng mắt (Error Analysis).
     * **`df_calib` (300 mẫu):** Dùng riêng để fit **Temperature Scaling ($T_{\text{sent}}, T_{\text{asp}}$)** và hiệu chỉnh ngưỡng **`aspect_threshold`**, **`low_conf_threshold`**.
     * **`df_blackbox` (562 mẫu):** Giữ sạch 100% chỉ để chấm điểm khách quan chọn mô hình Baseline / chọn checkpoint PhoBERT tốt nhất (không bị post-hoc overfit vì việc calibration đã tách sang `df_calib`).
   * **Baseline NB 03:** Dùng `sklearn.pipeline.Pipeline` + `StratifiedKFold(5)` thuần trên `df_train`. Chọn mô hình thắng thực tế trên `df_blackbox` (không ép cứng `LinearSVC` nếu `LogisticRegression` thắng).
5. **Transformer Engineering chuẩn mực (`vinai/phobert-base-v2`):** Giữ nguyên chữ Hoa/Thường (**cased**), bảo vệ placeholder PII khi chạy `pyvi`, đăng ký 5 special tokens vào `added_tokens` **và khởi tạo embedding bằng trung bình vector ngữ nghĩa tiếng Việt** (không để vector ngẫu nhiên!), chuyển emoji (kèm strip `\ufe0f`) sang từ tiếng Việt, `MAX_LEN = 256` dynamic padding, tách `weight_decay = 0.0` cho `bias` & `LayerNorm`, Early Stopping thật (`patience = 2` kèm `break`), và đánh giá độc lập từng head.
6. **XAI Trung thực & Gom cụm Ngữ nghĩa Thực sự:**
   * Global SHAP trên `lr_sentiment` chỉ tổng hợp khi từ **thực sự xuất hiện ($x_i > 0$)** theo `.mean(axis=0)` và lọc `present_count >= 5` + bổ sung **PhoBERT PartitionSHAP/Masking** trực tiếp trên `phobert_multitask.pt`.
   * Gom cụm Actionable Insights bằng **Embedding backbone `vinai/phobert-base-v2` gốc (chưa fine-tune ép 5 lớp)** + **`KMeans(k=6)`** (fit trên `Train` tiêu cực, predict trên `Test` tiêu cực) + **`c-TF-IDF`** khớp tên chủ đề động, xuất cột `sentence_clean` (đã che PII) vào `sample_reviews`.
7. **Universal Excel Ingestion 10/10 thật & Chống lộ PII:** Sửa `kw_bad` bằng so khớp chính xác / word-boundary (tránh `'time'` khớp `'Customer_Sentiment_Note'`), bắt `\t` cho cả UTF-8 TSV, đọc 4KB đầu khi sniff HTML-as-xls, xử lý `star_col is None`, tái tạo `05_pos_legacy.xls` (`xlwt`) và `09_merged_cells_pagination.xlsx` (`merge_cells` + lặp header thật), unmerge đúng tọa độ `ws.merged_cells.ranges`, che triệt để mọi cột tên khách hàng/SĐT/Mã đơn.

---

## 2. Thiết Kế Chi Tiết Từng Module & Lộ Trình Khắc Phục Lỗi

### 2.1. `src/labels.py` — Pure Sentiment & Multi-Label Aspect

* **`sentiment` (3 lớp: `0` Tiêu cực, `1` Trung lập/Hỗn hợp, `2` Tích cực) — Tuyệt đối không dùng `n_star`:**
  * Trích xuất tất cả cặp `{ASPECT#Polarity}` từ `label`. Nếu không có (ví dụ chỉ có `{OTHERS}`), trả về `sentiment = 1`.
  * Đếm `neg = sents.count(0)`, `neu = sents.count(1)`, `pos = sents.count(2)`.
  * Công thức **Majority-over-3** (khớp chính xác phân phối tự nhiên ~`32.4%` Tiêu cực / `19.6%` Trung lập / `48.1%` Tích cực trên Train, ~`19.5%` Trung lập trên Dev & Test):
    ```python
    if neg > pos and neg > neu:
        sent = 0
    elif pos > neg and pos > neu:
        sent = 2
    else:
        sent = 1
    ```
* **Multi-Label Aspect (`asp_0..asp_4` $\in \{0, 1\}$):**
  * Bật `1` cho **mọi** nhóm khía cạnh có mặt trong câu (`{OTHERS}` $\rightarrow$ `asp_4 = 1`).
  * `primary_aspect` (`0..4`): Chuẩn hóa NFC trước khi tìm; dùng regex ranh giới từ `(?<!\w)(?:...)(?!\w)` cho `ASPECT_KWS` (tránh `'giá'` $\leftrightarrow$ `'đánh giá'`, `'hình'` $\leftrightarrow$ `'cấu hình'`, `'loa'` $\leftrightarrow$ `'úp load'`); ưu tiên khía cạnh cụ thể không phải `GENERAL/OTHERS` xuất hiện sớm nhất trong câu.

---

### 2.2. `src/text_cleaner.py` — Vá Toàn Bộ 6 Lỗi Regex, Emoji, `pyvi` & PII

Thứ tự thực thi nghiêm ngặt trong `clean_text(text, lowercase=False)`:
1. **Unicode NFC & Loại bỏ Variation Selector ngay đầu hàm:**
   * `t = unicodedata.normalize('NFC', str(text)).replace('\ufe0f', '').replace('\u200d', '')` (giải quyết 240 Train / 35 Dev / 56 Test dòng NFD và xử lý triệt để các emoji như `❤️ = \u2764\ufe0f`).
2. **Bảo tồn Hoa/Thường (Cased) & Chuẩn hóa Emoji rộng:**
   * Map bảng emoji cảm xúc đa dạng (bao gồm mặt cười/buồn/giận/khóc/tim/like/dislike/sao/cờ) sang token tiếng Việt có gạch dưới (` tức_giận `, ` thất_vọng `, ` hài_lòng `, ` không_hài_lòng `, ` yêu_thích `, ` buồn_bã `) rồi xóa các ký tự icon Unicode lạ còn sót (tránh `<unk>` trên ~5% câu).
3. **Ẩn danh PII chuẩn chặt chẽ (Case-insensitive, có ranh giới từ):**
   * **`_URL_`:** Bắt `https?://\S+|www\.\S+` và domain thương mại phổ biến (`(?<!\w)(?:thegioididong\.com|dienmayxanh\.com|fptshop\.com\.vn|shopee\.vn|tiki\.vn|lazada\.vn|tiktok\.com|cellphones\.com\.vn|viettelstore\.vn|hoanghamobile\.com|[\w-]+\.(?:com|vn|com\.vn))(?=\s|$|/)` — **loại `.net` đứng trần sau từ tiếng Việt** như `"chụp.net"` trừ khi có `http://`/`www.`!).
   * **`_EMAIL_`:** `[\w\.-]+@[\w\.-]+\.\w+`.
   * **`_ORDER_ID_`:** Bắt 2 dạng thật an toàn (không bao giờ bắt nhầm `"affordable"`):
     * Có tiền tố đơn hàng: `(?<![A-Za-zÀ-ỹ0-9])#?(?:dh|đh|ord|spx)[-_]?[0-9][0-9a-zA-Z_-]{3,}\b` (bắt được `ORD-2024-001`, `SPX_12345`, `#ĐH889911`).
     * Hoặc chuỗi mã đơn toàn số / mã vận đơn sàn dài 12–18 ký tự không bắt đầu bằng `0`: `(?<!\d)[1-9][0-9]{11,17}(?!\d)`.
   * **`_PHONE_`:** `(?<!\d)(?:\+84|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-46-9])(?:[\s\.\-]?\d){7}(?!\d)` (bắt `090-123-4567`, `090.123.4567`, `+84 912 345 678`, không bắt nhầm mã đơn dài).
   * **`_PERSON_`:** **Tuyệt đối không dùng regex viết thường `\b(anh|chị|tên)\s+[a-zà-ỹ]+\s+[a-zà-ỹ]+`** (vì sẽ ăn nhầm 155+ cụm như `"anh em dùng"`, `"anh nhân viên"`). Chỉ ẩn danh `[PERSON]` trong văn bản tự do khi gặp **Tên Riêng Viết Hoa** (tối thiểu 2–4 âm tiết viết hoa chữ cái đầu, thuộc họ phổ biến Việt Nam hoặc đứng ngay cạnh ngữ cảnh PII `_PHONE_`/`_ORDER_ID_` và không chứa các từ danh từ chung/thương hiệu như `Thế Giới Di Động`, `Điện Máy Xanh`, `Nhân Viên`, `Samsung`, `iPhone`, `Oppo`, `Xiaomi`, `Shopee`, `Lazada`), kết hợp che 100% các cột tên người trong bảng Excel.
4. **Bảo vệ tiếng cười & Rút gọn ký tự kéo dài không phá từ (Case-insensitive):**
   * Chuyển `(?<!\w)[kK]{3,}(?!\w)` $\rightarrow$ ` cười `, `(?<!\w)(?:ha){2,}h?(?!\w)` $\rightarrow$ ` cười ` **trước** bước rút gọn ký tự (chấm dứt `kkk` $\rightarrow$ `k` $\rightarrow$ `không`).
   * **Sửa lỗi `'quáaaa'` $\rightarrow$ `'quá'` và bảo vệ `'free'`/`'full'`/`'ưu'`:**
     * Bước 4a — Rút gọn nguyên âm tiếng Việt kéo dài về đúng nguyên âm có dấu gốc (`re.IGNORECASE`):
       * `([àáảãạăằắẳẵặâầấẩẫậÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ])[aăâAĂÂ]+` $\rightarrow$ `\1` (biến `'quáaaa'` $\rightarrow$ `'quá'`!).
       * `([èéẻẽẹêềếểễệÈÉẺẼẸÊỀẾỂỄỆ])[eêEÊ]+` $\rightarrow$ `\1`
       * `([òóỏõọôồốổỗộơờớởỡợÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ])[oôơOÔƠ]+` $\rightarrow$ `\1`
       * `([ùúủũụÙÚỦŨỤ])[uU]+` và `([ưừứửữựƯỪỨỬỮỰ])[ưƯ]+` $\rightarrow$ `\1` (**KHÔNG gộp `ư` với `u`**, bảo toàn 100% `ưu`, `lưu`, `hữu`, `cứu`, `sưu`!).
       * `([ìíỉĩịÌÍỈĨỊ])[iI]+`, `([ỳýỷỹỵỲÝỶỸỴ])[yY]+` $\rightarrow$ `\1`.
     * Bước 4b — Với chữ lặp 3+ lần liên tiếp (vd `đẹppp`, `quáaaa` nếu toàn không dấu `quaAAA`, `freee`, `goood`):
       * Tập hợp whitelist từ có 2 chữ cái gốc giống nhau (`free`, `good`, `cool`, `full`, `app`, `call`, `mess`, `pass`, `boss`, `look`, `book`, `feel`, `need`, `keep`, `seed`, `week`, `zoom`, `room`, `moon`, `soon`, `oppo`, `google`, `shopee`, `tgdd`, `ss`, `facebook`, `xoong`, `boong`). Nếu từ hạ chữ lặp $\ge 3$ thuộc nhóm có 2 chữ gốc (ví dụ `freee` $\rightarrow$ `free`, `goood` $\rightarrow$ `good`), giữ nguyên 2 chữ; còn lại gộp `\1{2,}` $\rightarrow$ `\1` (**tuyệt đối không gộp cặp 2 chữ như `full`, `app`, `tgdd`**!).
5. **Teencode chuẩn hóa theo ngữ cảnh & ranh giới khoảng trắng:**
   * Giữ case hợp lý khi thay thế. Xử lý `m.n` $\rightarrow$ `mọi người`, `:v` $\rightarrow$ `haha`.
   * Chỉ thay từ viết tắt 1 chữ cái (`k`, `m`, `v`, `r`, `j`, `z`) khi đứng biệt lập giữa khoảng trắng (không dính dấu `.` hay `:`).
   * Thay `(?<!màu\s)(?<!Màu\s)\bcam\b(?!\s+ứng)` $\rightarrow$ `camera` (an toàn cho `"màu cam"`, `"cảm ứng"` / `"cam ứng"`).
6. **Khôi phục 5 Placeholder PII & Chạy `segment_words()` an toàn với `pyvi`:**
   * Khôi phục `_URL_ -> [URL]`, `_EMAIL_ -> [EMAIL]`, `_ORDER_ID_ -> [ORDER_ID]`, `_PHONE_ -> [PHONE]`, `_PERSON_ -> [PERSON]`.
   * Trong `segment_words(cleaned_text)`: chạy `ViTokenizer.tokenize(cleaned_text)`, **sau đó lập tức vá lại khoảng trắng do `pyvi` chèn quanh dấu ngoặc vuông**:
     ```python
     tok = re.sub(r'\[\s*(URL|EMAIL|ORDER_ID|PHONE|PERSON)\s*\]', r'[\1]', tok)
     ```
     (Khắc phục triệt để lỗi `pyvi` tách `[PHONE]` $\rightarrow$ `[ PHONE ]` đã kiểm chứng thực tế!).
7. **Bộ lọc `is_no_content` hoạt động thật:** Trả về `True` khi văn bản sạch `< 2` từ có chữ cái (sau khi bỏ `[PHONE]`, `[ORDER_ID]`...), hoặc độ dài $\le 8$ từ mà chứa cụm spam (`nhận xu`, `lấy xu`, `cho 5 sao`, `chưa dùng`, `chưa thử`, `không có ý kiến`, `đã nhận hàng`).

---

### 2.3. Chống Trùng Lặp Dữ Liệu (Deduplication) & Giảm Tải `Blackbox Dev`

1. **Deduplicate sau bước NB 01 (`01_text_cleaning_and_pii.ipynb`):**
   * Loại bỏ các câu trùng `sentence_clean` nội bộ trong `train_clean.csv` (giữ bản xuất hiện đầu tiên hoặc hợp nhất nhãn nếu trùng hoàn toàn).
   * Kiểm tra và xử lý các câu trong `train_clean.csv` bị trùng khít với `dev_clean.csv` hoặc `test_clean.csv` (xóa khỏi `train_clean` để **đảm bảo 0% overlap giữa Train và Dev/Test**, đồng thời kiểm tra xóa trùng mâu thuẫn nội bộ Dev).
2. **Phân tách `Dev` (`1,112` mẫu) thành 3 tập con độc lập:**
   * **`df_eyeball` (250 mẫu):** Soi lỗi thủ công giữa Baseline và PhoBERT.
   * **`df_calib` (300 mẫu):** Fit Temperature Scaling ($T_{\text{sent}}, T_{\text{asp}}$), chọn `aspect_threshold` tối ưu và chọn `low_conf_threshold` cho `needs_review`.
   * **`df_blackbox` (562 mẫu):** Giữ sạch hoàn toàn khỏi Error Analysis và khỏi bước Calibration, chỉ dùng duy nhất cho Early Stopping / chọn mô hình chiến thắng.

---

### 2.4. Khởi tạo Embedding & Huấn luyện Chuẩn Cho PhoBERT (NB 04 — Colab T4 GPU)

* **Colab Sync chuẩn tuyệt đối:** Upload/Clone kèm thư mục `src/` lên Colab (NB 04 ở Cell 1 kiểm tra và `assert os.path.exists('src/text_cleaner.py')`, **không có hàm duplicate inline**).
* **Khởi tạo Embedding ngữ nghĩa cho 5 Special Tokens mới:**
  ```python
  special_toks = ['[PHONE]', '[ORDER_ID]', '[EMAIL]', '[URL]', '[PERSON]']
  tokenizer.add_special_tokens({'additional_special_tokens': special_toks})
  model.encoder.resize_token_embeddings(len(tokenizer))
  seed_phrases = {
      '[PHONE]': 'số điện_thoại',
      '[ORDER_ID]': 'mã đơn_hàng',
      '[EMAIL]': 'email',
      '[URL]': 'đường_dẫn',
      '[PERSON]': 'khách_hàng',
  }
  # Gán vector khởi tạo bằng trung bình embedding của từ tương đương trong vocab gốc
  ```
  Nhờ vậy dù `[EMAIL]` hay `[ORDER_ID]` ít xuất hiện trong `Train.csv`, PhoBERT vẫn hiểu ngay ngữ nghĩa khi gặp dữ liệu Excel thực tế ở NB 06!
* **Kiến trúc Multi-Task & Tối ưu:**
  * `sent_head`: `Linear(768, 3)` + `CrossEntropyLoss(weight=w_s, label_smoothing=0.05)`.
  * `asp_head`: `Linear(768, 5)` + `BCEWithLogitsLoss(pos_weight=pos_w_a)` (**Multi-Label 5 khía cạnh**).
  * Dynamic padding theo từng batch (`MAX_LEN = 256`), AdamW tách `weight_decay = 0.0` cho `bias` & `LayerNorm`, `EPOCHS = 6`, **`EarlyStopping(patience=2)` có `break` thật**, theo dõi riêng `blackbox_sent_f1` & `blackbox_asp_f1` (lưu `best_sent_state` / `best_asp_state` hoặc balanced checkpoint khi cả 2 head hội tụ tốt nhất, ghi log trung thực vào `training_history.json`).
  * **Temperature Scaling + Ngưỡng `needs_review` trên `df_calib`:** Tối ưu nhiệt độ $T_{\text{sent}}, T_{\text{asp}}$ trên `df_calib`, lưu `calibration.json`; lưu ý kiểm tra cẩn thận điều kiện `star_contra` (đặc biệt cân nhắc tách riêng cảnh báo `4 sao + chê` ở mức nhẹ hoặc giữ chuẩn theo `Spec.md` và đo False Positive Rate thực tế trên `df_calib`).
* **Zip Colab xuất ra 100% đồng bộ:** `phobert_multitask.pt`, `training_history.json`, `calibration.json`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, `dev_pred_phobert.csv`, `test_pred_phobert.csv`, `model_metadata.json`.

---

### 2.5. Sửa Triệt Để NB 05 (XAI & Clustering) & NB 06 (Excel Ingestion 10/10)

1. **NB 05 (`05_shap_and_actionable_insights.ipynb`):**
   * **Global Lexical SHAP (`lr_sentiment`):** Lọc `present_mask = (X_te_arr > 0)`, tính `active_neg = np.where(present_mask, np.maximum(neg_shap, 0), 0.0)`, tính `active_neg.mean(axis=0)` và chỉ lấy các token có `present_mask.sum(axis=0) >= 5` $\rightarrow$ Chấm dứt hoàn toàn từ khen/từ hiếm 1 lần leo top!
   * **PhoBERT Local XAI:** Dùng `shap.Explainer(phobert_predict_proba, masker=shap.maskers.Text(tokenizer))` (PartitionSHAP / occlusion) giải thích trực tiếp dự đoán của `phobert_multitask.pt` trên các câu mẫu thực tế (gồm cả câu Baseline sai mà PhoBERT đúng).
   * **Actionable Insights Clustering thật:** Trích xuất **mean-pooled embedding từ backbone `vinai/phobert-base-v2` gốc** (không dùng `[CLS]` đã ép 5 lớp của mô hình fine-tuned) cho toàn bộ câu tiêu cực tập `Train`, chạy **`KMeans(n_clusters=6, random_state=42, n_init=20)`** (so Silhouette cho `k in 4..8`), dùng **`c-TF-IDF`** trích xuất Top 10 từ khóa đặc trưng cho mỗi cụm, gán tên chủ đề & khuyến nghị **động theo từ khóa c-TF-IDF khớp tốt nhất** (không hardcode `cluster_id`), dự đoán sang câu tiêu cực tập `Test`, và lưu `sample_reviews` từ cột **`sentence_clean` (đã che PII)**.
2. **NB 06 (`src/excel_ingestion.py` + `06_excel_auto_detector.ipynb` + `data/samples_messy/`):**
   * Tái tạo `05_pos_legacy.xls` bằng `xlwt` (BIFF nhị phân thật) và `09_merged_cells_pagination.xlsx` bằng `openpyxl` (`ws.merge_cells` + dòng tiêu đề lặp ở dòng giữa bảng).
   * `kw_bad`: Chuyển từ kiểm tra chuỗi con `b in c_low` sang kiểm tra **tách từ cột bằng `_` / khoảng trắng** (`tokens = set(re.split(r'[\s_]+', c_low))`, ví dụ `'id' in tokens`, `'time' in tokens`, `'sao' in tokens`, `'name' in tokens`, `'username' in tokens`, và **bỏ hẳn `'review_text'`, `'combined_review'`** khỏi `kw_bad` lúc detect raw).
   * `read_xls_safe`: Đọc **4,096 bytes đầu** để kiểm tra `<html`/`<table`/`<!doctype`.
   * `read_csv_safe`: Tự động phát hiện phân cách `\t`, `;`, `,` độc lập với bảng mã (bắt được cả **UTF-8 TSV** lẫn UTF-16 TSV).
   * Unmerge chính xác theo `ws.merged_cells.ranges` khi đọc `.xlsx` (không `.ffill()` mù cột SĐT/Số sao).
   * Xử lý fallback khi file upload **không có cột số sao (`star_col is None`)** (`Can_Kiem_Duyet` chỉ dựa vào `low_conf | is_no_content`).
   * Che PII toàn diện tất cả các cột nhạy cảm (`Tên người mua`, `Khách hàng`, `Buyer`, `Username` $\rightarrow$ `[PERSON]`, xóa/ẩn cột text thô gốc trong `sample_excel_predicted.xlsx` chỉ giữ cột đã che PII), và suy luận file mẫu 25 dòng bằng **PhoBERT Multi-Task** (kèm fallback `argmax` nếu Multi-Label `< threshold`).

---

## 3. Kế Hoạch Cập Nhật Đồng Bộ Toàn Dự Án (Bước 4 Sau Khi Train Colab)

1. Cập nhật lại toàn bộ tài liệu & cấu hình dự án cho khớp 100%:
   * [Spec.md](file:///d:/E-commerce/Spec.md) (bỏ đường dẫn lỗi thời `d:\What-If Simulator\coding_style.md`, cập nhật kiến trúc `src/` + Multi-Label Aspect + Pure Sentiment + KMeans/c-TF-IDF).
   * [requirements.txt](file:///d:/E-commerce/requirements.txt) (bổ sung đầy đủ `pyvi`, `accelerate`, `chardet`, `joblib`, `scipy`, `pytest`, `Pillow`, `xlwt`).
   * [coding_style.md](file:///d:/E-commerce/coding_style.md) (cập nhật ghi chú cho phép dùng `src/` & `sklearn.pipeline.Pipeline` chuẩn kỹ thuật).
   * [TAI_LIEU_BAN_GIAO_APP.md](file:///d:/E-commerce/TAI_LIEU_BAN_GIAO_APP.md) lên **v3.0** (đồng bộ số liệu, schema Multi-Label Aspect, calibration, và import trực tiếp từ `src/`).
2. Xóa bỏ các script vá cell lỗi thời trong `scripts/` (`update_and_run_nb01.py`, `update_and_run_notebook_03.py`, `verify_task1.py`) và nâng cấp toàn bộ `tests/` thành bộ test **chạy thực thi thật end-to-end** trên `src/`, NB00–06 và 10/10 file `samples_messy/`.
