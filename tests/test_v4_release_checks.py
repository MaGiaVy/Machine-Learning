"""Kiểm tra phát hành v4.1: .gitignore, fingerprint, needs_review, NB04/05/06 và số liệu README."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import review_exact_match

ROOT = Path(__file__).resolve().parents[1]


def _load_meta() -> dict:
    return json.loads((ROOT / 'models' / 'model_metadata.json').read_text(encoding='utf-8'))


def test_gitignore_blocks_heavy_artifacts():
    text = (ROOT / '.gitignore').read_text(encoding='utf-8')
    for pat in ['*.pt', 'phobert/', 'phobert_absa_eval_v4*.zip', '.tmp_run/']:
        assert pat in text, f'.gitignore thiếu {pat}'


def test_train_clean_fingerprint_and_split_sizes():
    proc = ROOT / 'data' / 'processed'
    tr = pd.read_csv(proc / 'train_clean.csv')
    dv = pd.read_csv(proc / 'dev_clean.csv')
    te = pd.read_csv(proc / 'test_clean.csv')
    assert (len(tr), len(dv), len(te)) == (7762, 1110, 2224)
    fp = hashlib.sha256(','.join(map(str, sorted(tr['index']))).encode()).hexdigest()[:16]
    assert fp == 'ad1b0833a88851e2'
    assert not (set(dv['sentence_clean']) & set(te['sentence_clean']))


def test_needs_review_rate_matches_budget_and_metadata():
    meta = _load_meta()
    info = meta['needs_review_test']
    te = pd.read_csv(ROOT / 'data' / 'processed' / 'test_pred_phobert.csv')
    rate = float(te['needs_review'].astype(bool).mean())
    assert abs(rate - info['test_flag_rate']) < 1e-3
    assert abs(rate - info['review_budget']) < 0.03
    cal = json.loads((ROOT / 'models' / 'phobert_absa_v4' / 'calibration.json').read_text(encoding='utf-8'))
    assert cal['review_conf_threshold'] == info['review_conf_threshold']


def test_review_exact_match_and_consistency():
    y = np.array([[0, 1, 0, 0, 3], [3, 3, 3, 0, 0]])
    p = np.array([[0, 1, 0, 0, 3], [3, 3, 2, 0, 0]])
    assert review_exact_match(y, p) == 0.5
    te = pd.read_csv(ROOT / 'data' / 'processed' / 'test_pred_phobert.csv')
    yt = te[[f'absa_{j}' for j in range(5)]].to_numpy()
    yp = te[[f'pred_absa_{j}' for j in range(5)]].to_numpy()
    exact = review_exact_match(yt, yp)
    err = _load_meta()['needs_review_test']['test_error_rate']
    assert abs((1.0 - err) - exact) < 0.002


def test_nb04_has_error_analysis_conclusion_and_nb06_no_legacy_key():
    nb4 = json.loads((ROOT / 'notebooks' / '04_phobert_multitask.ipynb').read_text(encoding='utf-8'))
    md = [''.join(c['source']) for c in nb4['cells'] if c['cell_type'] == 'markdown']
    hit = [t for t in md if t.lstrip().startswith('### Kết luận phân tích lỗi')]
    assert hit and len(hit[0].splitlines()) >= 4
    nb6_txt = (ROOT / 'notebooks' / '06_excel_auto_detector.ipynb').read_text(encoding='utf-8')
    assert 'low_conf_threshold' not in nb6_txt


def test_sample_excel_has_mixed_state_and_aspect_review_columns():
    df = pd.read_excel(ROOT / 'data' / 'processed' / 'sample_excel_predicted.xlsx', keep_default_na=False)
    for col in ['Du_Doan_Cam_Xuc', 'Khia_Canh_Can_Kiem', 'Muc_Do_Khieu_Nai_Ngam', 'Model_Su_Dung']:
        assert col in df.columns, col
    assert set(df['Muc_Do_Khieu_Nai_Ngam']) <= {'N/A', 'Không', 'Nhẹ', 'Cao'}
    assert 'Hỗn hợp' in set(df['Du_Doan_Cam_Xuc'])


def test_actionable_keywords_unique_across_clusters_and_no_fragments():
    ins = pd.read_csv(ROOT / 'data' / 'processed' / 'actionable_insights.csv')
    seen = {}
    for _, r in ins.iterrows():
        for kw in [k.strip() for k in str(r['extracted_keywords']).split(',') if k.strip()]:
            assert kw not in seen, f"'{kw}' lặp ở cụm {seen[kw]} và {r['cluster_id']}"
            seen[kw] = r['cluster_id']
    assert not ({'bảo', 'triệu', 'nghìn', 'đồng', 'lúc'} & set(seen))


def test_readme_numbers_match_metadata_and_limitations():
    meta = _load_meta()
    ph = meta['phobert_absa_v4']
    rv = meta['needs_review_test']
    ci = ph['paired_bootstrap_ci_vs_baseline']['diff_ci95']
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for s in [
        f"{ph['absa_macro_f1']:.4f}",
        f"{ph['mean_4cls_macro_f1']:.4f}",
        f"{ci[0]:.4f}",
        f"{ci[1]:.4f}",
        f"{rv['test_flag_rate'] * 100:.1f}%",
        f"{1.0 - rv['test_error_rate']:.4f}",
    ]:
        assert s in readme, f'README thiếu hoặc lệch số {s}'
    low = readme.lower()
    assert '## hạn chế' in low and '## tải weights' in low
    for kw in ['synthetic', 'seed', 'surrogate', 'neutral', 'dev_pred_phobert']:
        assert kw in low, f'README thiếu nội dung về {kw}'
