# VoC-Insight v3.1 Architecture Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the entire VoC-Insight pipeline (Notebooks 00–06, `src/` shared package, `data/samples_messy/`, models, and documentation) to eliminate all target leakage (`n_star`), validation leakage (`Blackbox Dev`), regex word destruction, SHAP sign inversion, fake clustering, and split-brain artifacts.

**Architecture:** Extract all label parsing, text normalization/PII masking/pyvi segmentation, Excel ingestion, and PyTorch/calibration logic into a single shared package `src/` imported identically by Local Notebooks, Colab NB 04, Pytest, and the FastAPI backend. Separate `Dev` into `Eyeball (250)` / `Calib (300)` / `Blackbox (562)` splits, tune Baseline models strictly via `StratifiedKFold(5)` on `Train` inside `sklearn.pipeline.Pipeline`, retrain `vinai/phobert-base-v2` (cased, semantic-initialized PII tokens, Pure Sentiment + Multi-Label Aspect, dynamic padding `MAX_LEN=256`, real `EarlyStopping`) once on Colab T4 GPU, and implement honest active-token/PhoBERT SHAP plus backbone KMeans + c-TF-IDF clustering.

**Tech Stack:** Python 3.12, Pandas, NumPy, Scikit-learn, PyVi, PyTorch, Transformers (`vinai/phobert-base-v2`), SHAP, OpenPyXL, xlrd, xlwt, chardet, pytest.

**Spec:** [2026-11-11-voc-insight-v3-architecture-design.md](file:///d:/E-commerce/docs/superpowers/specs/2026-11-11-voc-insight-v3-architecture-design.md)

## Global Constraints

- **Disk Space Workaround (Critical on this Host):** Drive `C:` currently has `0 bytes` free; `pyvi` (`sklearn_crfsuite`) and `nbconvert` fail if `%TEMP%` points to `C:`. Whenever running Python/`pytest`/`jupyter` locally, set `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"` (creating it before and cleaning it up or ignoring `.tmp_run/` in `.gitignore`).
- **Single Source of Truth:** NEVER duplicate `parse_visfd_labels`, `clean_text`, `segment_words`, or `read_any_file` across notebooks; all notebooks (Local and Colab) and tests must import from `src/`.
- **Notebook Style:** Keep notebook cells focused and readable with immediate `.shape`, `.head()`, `.value_counts()` verification cells and short Vietnamese section headers/comments; `random_state=42` everywhere.
- **Zero Target/Validation Leakage:** `n_star` must NEVER enter `sentiment` ground truth; `df_blackbox` must NEVER be used in CV hyperparameter tuning or temperature/threshold fitting.

## Review Focus

- `pyvi.ViTokenizer.tokenize` inserts spaces around brackets (`[ PHONE ]`), breaking PhoBERT `added_tokens` unless `segment_words()` immediately restores `r'\[\s*(URL|EMAIL|ORDER_ID|PHONE|PERSON)\s*\]'` $\rightarrow$ `r'[\1]'`.
- Elongated character collapsing turns `'quáaaa'` into `'quáa'` and `'freee'` into `'fre'` unless vowel-family rules (`[àáảãạăằắẳẵặâầấẩẫậ][aăâ]+` excluding `ư` before `u`) and English double-letter whitelists run before `\1{2,}`.
- Substring blacklisting in Excel detection (`'time' in 'customer_sentiment_note'`, `'review_text' in 'review_text'`) silently drops valid review columns unless column tokens are split on whitespace/underscores.
- Global `LinearExplainer` assigns positive SHAP values to *absent* positive words (`ok`, `tốt`, `đẹp`) unless masked to `X > 0` with `present_count >= 5`.
- Multi-label Aspect output where all 5 sigmoid probabilities fall below `aspect_threshold` leaves `primary_aspect` undefined unless an `argmax` fallback selects the highest-probability group.

---

### Task 1: Build Shared Core Package (`src/labels.py` & `src/text_cleaner.py`) with Strict Unit Tests

**Files:**
- Create: `src/__init__.py`, `src/labels.py`, `src/text_cleaner.py`
- Test: `tests/test_src_core.py`
- Modify: `.gitignore` (add `.tmp_run/`)

**Interfaces:**
- Produces:
  - `src.labels.ASPECT_GROUPS: list[str]` (`['PERF_FEAT', 'SCREEN_CAM', 'BATTERY', 'PRICE_SER', 'DESIGN_GEN']`)
  - `src.labels.parse_visfd_labels(label_str: str, comment: str) -> dict` returning `{'sentiment': int, 'aspect_group': int, 'aspects': list[int], 'all_aspects': list[int]}` (`aspects` is length-5 binary list `[0/1]*5`)
  - `src.text_cleaner.clean_text(text: str, lowercase: bool = False) -> str`
  - `src.text_cleaner.check_no_content(cleaned_text: str) -> bool`
  - `src.text_cleaner.segment_words(cleaned_text: str) -> str`

- [ ] **Step 1: Write failing unit tests in `tests/test_src_core.py`**

```python
import os, tempfile
os.environ['TMP'] = os.environ['TEMP'] = 'D:/E-commerce/.tmp_run'
os.makedirs('D:/E-commerce/.tmp_run', exist_ok=True)
tempfile.tempdir = 'D:/E-commerce/.tmp_run'

from src.labels import parse_visfd_labels
from src.text_cleaner import clean_text, check_no_content, segment_words

def test_pure_sentiment_no_star_leakage_and_multilabel():
    # Pure majority-over-3 without n_star; Pin kém + Tổng quan miễn chê -> both BATTERY(2) and DESIGN_GEN(4) active
    res = parse_visfd_labels("{BATTERY#Negative};{DESIGN#Positive};{GENERAL#Positive};", "Pin kém còn lại miễn chê")
    assert res["sentiment"] == 2
    assert res["aspects"] == [0, 0, 1, 0, 1]
    assert res["aspect_group"] == 2  # BATTERY found at index 0 via \b keyword 'pin'
    # Boundary check: 'đánh giá' must NOT trigger PRICE_SER(3) and 'úp load' must NOT trigger PERF_FEAT(0)
    res2 = parse_visfd_labels("{PRICE#Neutral};{PERFORMANCE#Neutral};", "đánh giá sơ bộ úp load hình pin")
    assert res2["sentiment"] == 1
    assert res2["aspects"] == [1, 0, 0, 1, 0]

def test_text_cleaner_and_pyvi_placeholder_protection():
    raw = "TGDĐ đt quáaaa đẹppp, freee full app call, lưu ý kkk màu cam affordable ORD-2024-001 sđt 090-123-4567 tại thegioididong.com chụp.net ❤️ anh nhân viên"
    cleaned = clean_text(raw, lowercase=False)
    assert "quá" in cleaned and "quáa" not in cleaned
    assert "free full app call" in cleaned
    assert "lưu ý" in cleaned and "lư ý" not in cleaned
    assert "cười" in cleaned and "không" not in cleaned
    assert "màu cam" in cleaned and "màu camera" not in cleaned
    assert "affordable" in cleaned and "af [ORDER_ID]" not in cleaned
    assert "[ORDER_ID]" in cleaned and "[PHONE]" in cleaned and "[URL]" in cleaned
    assert "chụp.net" in cleaned and "anh nhân viên" in cleaned
    seg = segment_words(cleaned)
    for tok in ["[ORDER_ID]", "[PHONE]", "[URL]"]:
        assert tok in seg and f"[ {tok[1:-1]} ]" not in seg
    assert check_no_content("cho 5 sao nhận xu") is True
    assert check_no_content("Máy dùng tốt") is False
```

- [ ] **Step 2: Run pytest to verify FAIL (`ModuleNotFoundError: No module named 'src'`)**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_src_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Implement `src/__init__.py`, `src/labels.py`, and `src/text_cleaner.py`**
  - Add `.tmp_run/` to `.gitignore`.
  - In `src/labels.py`: implement `parse_visfd_labels(label_str, comment)` with Majority-over-3 sentiment (`neg > pos and neg > neu -> 0`, `pos > neg and pos > neu -> 2`, else `1`), binary 5-element vector `aspects` (`{OTHERS}` sets `aspects[4] = 1`), Unicode NFC normalization before keyword lookup, and `(?<!\w)(?:...)(?!\w)` word-boundary regexes for `primary_aspect`.
  - In `src/text_cleaner.py`: implement exact step order: NFC + strip `\ufe0f`/`\u200d` $\rightarrow$ emoji mapping $\rightarrow$ PII masking (`_URL_`, `_EMAIL_`, `_ORDER_ID_`, `_PHONE_`, capitalized-only `_PERSON_` excluding brand/role words) $\rightarrow$ `[kK]{3,}`/`(ha){2,}` $\rightarrow$ `cười` $\rightarrow$ vowel family collapse (preserving `ưu`) + 3+ char collapse preserving English double-letter whitelist (`free`, `full`, `app`, `call`, `mess`, `good`, `tgdd`, `ss`, `shopee`, `oppo`...) $\rightarrow$ boundary-safe teencode (`m.n`, `:v`, contextual `cam`) $\rightarrow$ restore `[URL]`/`[EMAIL]`/`[ORDER_ID]`/`[PHONE]`/`[PERSON]` $\rightarrow$ `segment_words()` calling `ViTokenizer.tokenize` and immediately repairing bracket spacing `r'\[\s*(URL|EMAIL|ORDER_ID|PHONE|PERSON)\s*\]' -> r'[\1]'`.

- [ ] **Step 4: Run pytest to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_src_core.py -v`
Expected: PASS (2/2)

---

### Task 2: Refactor & Execute NB 00 (Pure Labels + Merged File), NB 01 (Cleaning + Cross-Split Deduplication), & NB 02 (BPE Subwords EDA)

**Files:**
- Modify: `notebooks/00_load_and_merge_raw.ipynb`, `notebooks/01_text_cleaning_and_pii.ipynb`, `notebooks/02_eda.ipynb`
- Output: `data/processed/train_raw.csv`, `dev_raw.csv`, `test_raw.csv`, `visfd_merged_raw.csv`, `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`, `reports/figures/*.png`
- Test: `tests/test_task1.py`

**Interfaces:**
- Consumes: `src.labels.parse_visfd_labels`, `src.text_cleaner.{clean_text, check_no_content, segment_words}`, `data/raw/{Train,Dev,Test}.csv`
- Produces: Deduplicated `train_clean.csv`, `dev_clean.csv`, `test_clean.csv` with `0` cross-split `sentence_clean` overlaps, binary columns `asp_0..asp_4`, cased `sentence_clean` & `sentence_seg`, and updated EDA figures + PhoBERT BPE length quantiles.

- [ ] **Step 1: Update `tests/test_task1.py` with v3.1 data integrity assertions**

```python
def test_v3_pure_labels_dedup_and_clean_integrity():
    tr = pd.read_csv('data/processed/train_clean.csv')
    de = pd.read_csv('data/processed/dev_clean.csv')
    te = pd.read_csv('data/processed/test_clean.csv')
    assert os.path.exists('data/processed/visfd_merged_raw.csv')
    # No n_star leakage: must contain 4-5 star Negative OR 1-2 star Positive reviews in ground truth
    true_contra = ((te['n_star'] <= 2) & (te['sentiment'] == 2)) | ((te['n_star'] >= 4) & (te['sentiment'] == 0))
    assert true_contra.sum() >= 50, f"Expected >= 50 real star contradictions in Test, got {true_contra.sum()}"
    # Multi-label columns present
    for c in ['asp_0', 'asp_1', 'asp_2', 'asp_3', 'asp_4']:
        assert c in tr.columns and set(tr[c].unique()).issubset({0, 1})
    # Zero clean-text leakage across Train vs Dev/Test and zero internal duplicates in Train
    s_tr, s_de, s_te = set(tr['sentence_clean'].fillna('')), set(de['sentence_clean'].fillna('')), set(te['sentence_clean'].fillna(''))
    assert len(s_tr & s_de) == 0 and len(s_tr & s_te) == 0
    assert tr['sentence_clean'].duplicated().sum() == 0
    assert tr['is_no_content'].sum() > 0, "is_no_content filter should trigger on real empty/spam rows"
    assert not tr['sentence_clean'].fillna('').str.contains(r'\baf \[ORDER_ID\]\b').any()
```

- [ ] **Step 2: Run pytest to verify FAIL on current v2 datasets**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task1.py::test_v3_pure_labels_dedup_and_clean_integrity -v`
Expected: FAIL (`AssertionError: Expected >= 50 real star contradictions in Test, got 0`)

- [ ] **Step 3: Refactor & Execute NB 00, NB 01, and NB 02 in order**
  - **NB 00:** Import `src.labels.parse_visfd_labels`, generate `train_raw.csv`, `dev_raw.csv`, `test_raw.csv` (with `asp_0..asp_4` + `all_aspects` + `aspect_group` + pure `sentiment`), concatenate into `visfd_merged_raw.csv` with `split` column, execute in-place (`--execute --inplace`).
  - **NB 01:** Import `src.text_cleaner.{clean_text, check_no_content, segment_words}`, compute `sentence_clean`, `is_no_content`, `sentence_seg`, run **deduplication** (drop internal duplicates in `train_clean` and `dev_clean` conflicting pairs; drop any `train_clean` rows whose `sentence_clean` appears in `dev_clean` or `test_clean`), print before/after Token Diff summary (`ưu`, `full/app/call`, `kkk`, `[ORDER_ID]`, `is_no_content`), save CSVs, execute in-place.
  - **NB 02:** Execute **after** NB 01 on the updated `train_clean.csv`; load `AutoTokenizer.from_pretrained('vinai/phobert-base-v2')` (local `models/phobert_multitask` if cached) to compute BPE subword length percentiles (`p95`, `p99`, `max`) justifying `MAX_LEN = 256`; add clear `xlabel`/`ylabel` on `eda_aspect_dist.png` (ordered bar palette), plot row-normalized `%` in `eda_aspect_sentiment_heatmap.png`, plot true `n_star` vs `sentiment` (highlighting real contradictions), execute in-place.

- [ ] **Step 4: Run `pytest tests/test_task1.py tests/test_src_core.py -v` to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task1.py tests/test_src_core.py -v`
Expected: PASS

---

### Task 3: Refactor & Execute NB 03 (`03_baseline_tfidf_models.ipynb`) with Pure-Train `Pipeline`, `StratifiedKFold(5)`, & Multi-Label Baseline

**Files:**
- Create: `src/modeling.py` (shared 3-way Dev split helper `split_dev_three_way` + common `PhoBertMultiTask` & `TemperatureScaler` to be used in Task 5 as well)
- Modify: `notebooks/03_baseline_tfidf_models.ipynb`
- Output: `models/tfidf_vectorizer.pkl`, `models/baseline_sentiment.pkl`, `models/baseline_aspect_multilabel.pkl`, `models/lr_sentiment.pkl` (explicitly kept for SHAP lexical baseline), `models/svc_sentiment.pkl`, `data/processed/test_pred_baseline.csv`
- Test: `tests/test_task2.py`

**Interfaces:**
- Consumes: `data/processed/{train,dev,test}_clean.csv`
- Produces: `src.modeling.split_dev_three_way(df_dev)` returning `(df_eyeball_250, df_calib_300, df_blackbox_562)`; best tuned Sentiment & Multi-Label Aspect baseline pipelines; `test_pred_baseline.csv` containing `pred_sentiment` and multi-label `pred_asp_0..pred_asp_4` + `pred_aspect`.

- [ ] **Step 1: Update `tests/test_task2.py` to enforce `Pipeline`, `StratifiedKFold`, clean `Blackbox Dev`, and Multi-Label Baseline**

```python
def test_v3_baseline_pipeline_no_blackbox_leak_and_multilabel():
    with open('notebooks/03_baseline_tfidf_models.ipynb', encoding='utf-8') as f:
        src_txt = f.read()
    assert 'StratifiedKFold' in src_txt and 'Pipeline(' in src_txt
    assert 'PredefinedSplit' not in src_txt, "Must not tune hyperparameters on Blackbox Dev!"
    assert 'df_eyeball' in src_txt and 'df_calib' in src_txt and 'df_blackbox' in src_txt
    df_base = pd.read_csv('data/processed/test_pred_baseline.csv')
    for c in ['pred_asp_0', 'pred_asp_1', 'pred_asp_2', 'pred_asp_3', 'pred_asp_4', 'pred_aspect', 'pred_sentiment']:
        assert c in df_base.columns
```

- [ ] **Step 2: Run pytest to verify FAIL**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task2.py::test_v3_baseline_pipeline_no_blackbox_leak_and_multilabel -v`
Expected: FAIL

- [ ] **Step 3: Implement `split_dev_three_way` in `src/modeling.py` and refactor + execute `notebooks/03_baseline_tfidf_models.ipynb`**
  - `split_dev_three_way(df_dev, random_state=42)`: stratifies by `sentiment.astype(str) + "_" + aspect_group.astype(str)` (falling back to `sentiment` for strata with count $< 3$) to split `1,112` dev rows into `df_eyeball` (250), `df_calib` (300), `df_blackbox` (562).
  - In NB 03:
    - Tune `ComplementNB` (`alpha in [0.05, 0.1, 0.25, 0.5, 1.0], norm`), `LogisticRegression`, `LinearSVC` via `Pipeline([('tfidf', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b|\[\w+\]', ngram_range=(1,2), max_features=15000, min_df=2, sublinear_tf=True, lowercase=True)), ('clf', ...)])` + `StratifiedKFold(5, shuffle=True, random_state=42)` strictly on `df_train`.
    - Tune Multi-Label Aspect (`OneVsRestClassifier(LogisticRegression)` & `OneVsRestClassifier(LinearSVC)`) via `StratifiedKFold(5)` / CV on `df_train`.
    - Evaluate `Train F1`, `Eyeball Dev F1`, `Blackbox Dev F1`; automatically pick the true winner on `df_blackbox` (saving `baseline_sentiment.pkl` and `baseline_aspect_multilabel.pkl`, plus `tfidf_vectorizer.pkl` & `lr_sentiment.pkl` fitted on `df_train` for Task 6 SHAP).
    - Run Error Analysis on `df_eyeball` (250) and final evaluation on `df_test` (`Macro F1`, `Multi-label Aspect F1`, `Hit-Any-True-Group`), saving `test_pred_baseline.csv`. Execute NB 03 in-place.

- [ ] **Step 4: Run `pytest tests/test_task2.py -v` to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task2.py -v`
Expected: PASS

---

### Task 4: Build `src/excel_ingestion.py`, Regenerate Authentic `05_pos_legacy.xls` & `09_merged_cells_pagination.xlsx`, & Fix NB 06 Ingestion to True 10/10 PASS

**Files:**
- Create: `src/excel_ingestion.py`
- Modify: `data/samples_messy/05_pos_legacy.xls`, `data/samples_messy/09_merged_cells_pagination.xlsx`, `notebooks/06_excel_auto_detector.ipynb`
- Test: `tests/test_task6_excel_ingestion.py`

**Interfaces:**
- Consumes: `src.text_cleaner.{clean_text, check_no_content, segment_words}`
- Produces:
  - `src.excel_ingestion.read_any_file(file_path: str) -> tuple[dict[str, pd.DataFrame], dict]` (also returns OpenPyXL `merged_ranges` metadata for exact unmerging!)
  - `src.excel_ingestion.pick_best_sheet`, `detect_header_row`, `detect_text_columns`, `detect_star_column`, `build_clean_table`, `mask_tabular_pii`

- [ ] **Step 1: Write failing test in `tests/test_task6_excel_ingestion.py` checking authentic binary `.xls`, real merged cells, and 10/10 detection (including `07_utf16_tab_export.csv` and `Customer_Sentiment_Note`)**

```python
import os, openpyxl, pandas as pd
from src.excel_ingestion import read_any_file, pick_best_sheet, detect_header_row, detect_text_columns, detect_star_column, build_clean_table, mask_tabular_pii

def test_authentic_messy_files_and_10_of_10_ingestion():
    # 05_pos_legacy.xls must be real binary BIFF Excel, not HTML
    with open('data/samples_messy/05_pos_legacy.xls', 'rb') as f:
        assert f.read(8) == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', "05_pos_legacy.xls is not binary BIFF .xls"
    # 09_merged_cells_pagination.xlsx must contain real merged cells
    wb09 = openpyxl.load_workbook('data/samples_messy/09_merged_cells_pagination.xlsx')
    assert len(wb09.active.merged_cells.ranges) > 0, "09_merged_cells_pagination.xlsx has no real merged cells"
    # 'Customer_Sentiment_Note' must NOT be blocked by 'time' substring in kw_bad
    df_edge = pd.DataFrame({'Customer_Sentiment_Note': ['Pin rất tốt nhưng hơi nóng khi chơi game nặng' for _ in range(10)]})
    assert 'Customer_Sentiment_Note' in detect_text_columns(df_edge)
    # 07_utf16_tab_export.csv must detect Review_Text properly
    sheets07, meta07 = read_any_file('data/samples_messy/07_utf16_tab_export.csv')
    _, raw07 = pick_best_sheet(sheets07)
    tab07 = build_clean_table(raw07, detect_header_row(raw07), meta=meta07)
    assert 'Review_Text' in detect_text_columns(tab07)
```

- [ ] **Step 2: Run pytest to verify FAIL**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task6_excel_ingestion.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.excel_ingestion'`)

- [ ] **Step 3: Implement `src/excel_ingestion.py`, regenerate `05_pos_legacy.xls` (`xlwt`) and `09_merged_cells_pagination.xlsx` (`openpyxl` merged cells + repeated mid-page header), and update NB 06**
  - Ensure `read_xls_safe` sniffs 4,096 bytes for `<html`/`<table`/`<!doctype`; `read_csv_safe` checks `\t`, `;`, `,` on ANY encoding (catches UTF-8 TSV too).
  - Tokenize column names on `[\s_]+` for `KW_BAD_TOKENS` (`{'id', 'ma', 'mã', 'sdt', 'sđt', 'phone', 'sao', 'star', 'rating', 'name', 'ten', 'tên', 'username', 'ngay', 'ngày', 'date', 'time', 'stt'}`) and drop `'review_text'`/`'combined_review'` from `KW_BAD_TOKENS`.
  - Unmerge ONLY coordinates inside `ws.merged_cells.ranges` for `.xlsx` (no blind `.ffill()` across blank phone/rating cells!), drop repeated mid-page header rows, concatenate multi-text columns, handle `star_col is None`, and mask ALL sensitive columns (`Tên người mua`, `Khách hàng`, `Buyer`, `Username` $\rightarrow$ `[PERSON]`, `SĐT`/`Số điện thoại` $\rightarrow$ `[PHONE]`, `Mã đơn`/`Order` $\rightarrow$ `[ORDER_ID]`).
  - Update NB 06 to run 10/10 benchmark PASS (`10 / 10`), using Baseline for temporary sanity check if PhoBERT v3 `.pt` hasn't been trained on Colab yet (will be switched to PhoBERT v3 in Task 7).

- [ ] **Step 4: Run `pytest tests/test_task6_excel_ingestion.py -v` to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task6_excel_ingestion.py -v`
Expected: PASS

---

### Task 5: Complete `src/modeling.py` & Refactor NB 04 (`04_phobert_multitask.ipynb`) Ready for 1-Shot Google Colab T4 GPU Run

**Files:**
- Modify: `src/modeling.py`, `notebooks/04_phobert_multitask.ipynb`
- Test: `tests/test_task3_phobert_arch.py`

**Interfaces:**
- Consumes: `src.labels`, `src.text_cleaner`, `data/processed/{train,dev,test}_clean.csv`
- Produces:
  - `src.modeling.PhoBertMultiTask`: cased `vinai/phobert-base-v2`, semantic-initialized 5 PII tokens, `sent_head(3)` + `asp_head(5)`, `MAX_LEN=256`, dynamic padding collate_fn.
  - `src.modeling.TemperatureScaler` (optimizing $T_{\text{sent}}, T_{\text{asp}}$ on `df_calib` logits).
  - Clean, executable Colab-ready `notebooks/04_phobert_multitask.ipynb` importing strictly from `src/` (requiring `src/` uploaded/cloned alongside notebook on Colab — zero inline duplicate parsers!), real `EarlyStopping(patience=2)` with `break`, calibrated `needs_review`, exporting `dev_pred_phobert.csv`, `test_pred_phobert.csv`, `phobert_multitask.pt`, `calibration.json`, `training_history.json`, `model_metadata.json`, and updated tokenizer into `phobert_ecommerce_outputs_v3.zip`.

- [ ] **Step 1: Write unit + dry-run test in `tests/test_task3_phobert_arch.py`**

```python
import torch, json
from transformers import AutoConfig, AutoTokenizer
from src.modeling import PhoBertMultiTask, init_semantic_pii_embeddings, TemperatureScaler, dynamic_collate_fn

def test_phobert_v3_architecture_semantic_init_and_calibration():
    tok_dir = 'models/phobert_multitask'
    tok = AutoTokenizer.from_pretrained(tok_dir)
    specials = ['[PHONE]', '[ORDER_ID]', '[EMAIL]', '[URL]', '[PERSON]']
    tok.add_special_tokens({'additional_special_tokens': specials})
    cfg = AutoConfig.from_pretrained(tok_dir)
    model = PhoBertMultiTask(config=cfg, dropout=0.3)
    model.encoder.resize_token_embeddings(len(tok))
    init_semantic_pii_embeddings(model.encoder, tok)
    # Verify [PHONE] embedding equals mean of 'số điện_thoại' embeddings (not random!)
    emb_w = model.encoder.get_input_embeddings().weight.detach()
    ref_ids = tok.encode('số điện_thoại', add_special_tokens=False)
    assert torch.allclose(emb_w[tok.convert_tokens_to_ids('[PHONE]')], emb_w[ref_ids].mean(dim=0), atol=1e-5)
    # Verify forward pass produces (B, 3) and (B, 5) with dynamic padding
    batch = dynamic_collate_fn([
        {'input_ids': torch.tensor([0, 5, 2]), 'attention_mask': torch.tensor([1, 1, 1]), 'sentiment': torch.tensor(2), 'aspects': torch.tensor([1,0,0,0,1], dtype=torch.float)},
        {'input_ids': torch.tensor([0, 5, 6, 7, 2]), 'attention_mask': torch.tensor([1, 1, 1, 1, 1]), 'sentiment': torch.tensor(0), 'aspects': torch.tensor([0,1,0,0,0], dtype=torch.float)},
    ], pad_id=tok.pad_token_id)
    ls, la = model(batch['input_ids'], batch['attention_mask'])
    assert ls.shape == (2, 3) and la.shape == (2, 5)
    # Verify NB 04 uses src/ imports and has real EarlyStopping break
    with open('notebooks/04_phobert_multitask.ipynb', encoding='utf-8') as f:
        nb4_txt = f.read()
    assert 'parse_label_colab' not in nb4_txt and 'src.modeling' in nb4_txt and 'break' in nb4_txt
```

- [ ] **Step 2: Run pytest to verify FAIL**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task3_phobert_arch.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `PhoBertMultiTask`, `init_semantic_pii_embeddings`, `dynamic_collate_fn`, `TemperatureScaler` in `src/modeling.py` and rewrite `notebooks/04_phobert_multitask.ipynb`**
  - Verify AdamW optimizer parameter groups split `bias` / `LayerNorm.weight` to `weight_decay=0.0`.
  - Include clear 2-line Colab setup instructions at top of NB 04 (how to upload/unzip `src/` + `data/processed/` or clone GitHub repo on Colab).
  - Run a lightweight sanity check on CPU (`tests/test_task3_phobert_arch.py`) confirming zero syntax/runtime errors before handing off to user for Colab GPU training.

- [ ] **Step 4: Run `pytest tests/test_task3_phobert_arch.py -v` to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task3_phobert_arch.py -v`
Expected: PASS

---

### [HUMAN CHECKPOINT]: User Trains NB 04 on Google Colab T4 GPU
- Hand user exact 3-step guide (push or upload `src/` + `data/processed/` + `04_phobert_multitask.ipynb` to Colab T4 GPU $\rightarrow$ Run All $\rightarrow$ drop `phobert_ecommerce_outputs_v3.zip` into `d:\E-commerce\`).

---

### Task 6: Unpack Colab v3 Outputs, Fix & Execute NB 05 (Active-Only Global SHAP, PhoBERT PartitionSHAP, & KMeans + c-TF-IDF Clustering)

**Files:**
- Modify: `notebooks/05_shap_and_actionable_insights.ipynb`
- Output: `models/phobert_multitask/*`, `models/model_metadata.json`, `data/processed/{dev,test}_pred_phobert.csv`, `data/processed/actionable_insights.csv`, `reports/figures/shap_top_negative_words.png`
- Test: `tests/test_task4_xai_insights.py`

**Interfaces:**
- Consumes: Colab `phobert_ecommerce_outputs_v3.zip`, `models/lr_sentiment.pkl`, `models/tfidf_vectorizer.pkl`, `src/modeling.PhoBertMultiTask`
- Produces: Verified active-only negative SHAP chart (0 praise words!), PhoBERT local PartitionSHAP explanations, and KMeans(k=6) + c-TF-IDF `actionable_insights.csv` with PII-masked `sentence_clean` samples and 100% coverage of Test `pred_sentiment == 0`.

- [ ] **Step 1: Write test in `tests/test_task4_xai_insights.py`**

```python
import pandas as pd, json

def test_xai_no_praise_words_and_real_kmeans_insights():
    df_tp = pd.read_csv('data/processed/test_pred_phobert.csv')
    df_ins = pd.read_csv('data/processed/actionable_insights.csv')
    assert df_ins['negative_count'].sum() == int((df_tp['pred_sentiment'] == 0).sum())
    assert len(df_ins) == 6
    # Sample reviews in actionable_insights must be PII-masked (no 10-digit phones)
    assert not df_ins['sample_reviews'].fillna('').str.contains(r'\b0[0-9]{9}\b').any()
    with open('notebooks/05_shap_and_actionable_insights.ipynb', encoding='utf-8') as f:
        nb5_txt = f.read()
    assert 'KMeans' in nb5_txt and 'present_mask' in nb5_txt and 'PartitionExplainer' in nb5_txt
```

- [ ] **Step 2: Unpack `phobert_ecommerce_outputs_v3.zip`, refactor & execute `notebooks/05_shap_and_actionable_insights.ipynb` in-place**
  - Global SHAP on `lr_sentiment`: mask `present_mask = (X_te_arr > 0)`, average `np.where(present_mask, np.maximum(neg_shap, 0), 0.0).mean(axis=0)`, filter `present_mask.sum(axis=0) >= 5`, verify Top 15 words contain zero praise words (`ok`, `tốt`, `đẹp`, `mượt`, `ngon`, `ổn`, `nhiệt_tình`).
  - Local PhoBERT XAI: run `shap.Explainer(phobert_predict_proba, masker=shap.maskers.Text(tokenizer))` on 3 real test cases (including a case where Baseline missed and PhoBERT succeeded).
  - Real clustering: compute mean-pooled embeddings using **pre-trained `vinai/phobert-base-v2` backbone** on `df_train[sentiment == 0]`, evaluate Silhouette for `k in range(4, 9)`, fit `KMeans(n_clusters=6, random_state=42, n_init=20)`, extract `c-TF-IDF` top keywords, map cluster names dynamically by `c-TF-IDF` keyword score, predict clusters on `df_test_pred[pred_sentiment == 0]`, store PII-masked `sentence_clean` samples into `actionable_insights.csv`.

- [ ] **Step 3: Run `pytest tests/test_task4_xai_insights.py -v` to verify PASS**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/test_task4_xai_insights.py -v`
Expected: PASS

---

### Task 7: Finalize NB 06 with PhoBERT v3 Inference, Sync All Project Docs, Clean Legacy Patch Scripts, & Run Full End-to-End Verification

**Files:**
- Modify: `notebooks/06_excel_auto_detector.ipynb`, `Spec.md`, `requirements.txt`, `coding_style.md`, `TAI_LIEU_BAN_GIAO_APP.md`
- Delete: `scripts/update_and_run_nb01.py`, `scripts/update_and_run_notebook_03.py`, `scripts/verify_task1.py` (plus old root dump helper scripts `dump_*.py` and `check_notebooks.py` if confirmed unused)
- Test: `tests/verify_fixes.py` (end-to-end verification across all 7 notebooks, models, 10 messy files, and `sample_excel_predicted.xlsx`)

- [ ] **Step 1: Execute `notebooks/06_excel_auto_detector.ipynb` using `PhoBertMultiTask` + `calibration.json` on `01_shopee_clean.xlsx` to produce final `data/processed/sample_excel_predicted.xlsx` (verifying 10/10 messy file benchmark PASS and 0 leaked PII columns including buyer names).**
- [ ] **Step 2: Delete obsolete patch scripts in `scripts/`, update `Spec.md`, `requirements.txt`, `coding_style.md`, and `TAI_LIEU_BAN_GIAO_APP.md` (v3.0) with exact finalized metrics, multi-label schema, calibration, and `src/` imports.**
- [ ] **Step 3: Run full test suite & verify 100% PASS with real end-to-end checks**

Run: `$env:TEMP = $env:TMP = "D:\E-commerce\.tmp_run"; pytest tests/ -v`
Expected: PASS (100% across all test files)
