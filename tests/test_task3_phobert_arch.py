import ast
import json
import os
import sys
import tempfile

os.environ['TMP'] = os.environ['TEMP'] = 'D:/E-commerce/.tmp_run'
os.environ['HF_HOME'] = 'D:/E-commerce/.tmp_run/hf'
os.environ['HF_HUB_OFFLINE'] = '1'
os.makedirs('D:/E-commerce/.tmp_run', exist_ok=True)
tempfile.tempdir = 'D:/E-commerce/.tmp_run'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer

from src.modeling import (
    MAX_LEN,
    PII_SEED_PHRASES,
    PII_TOKENS,
    PhoBertMultiTask,
    TemperatureScaler,
    ViSFDDataset,
    dynamic_collate_fn,
    init_semantic_pii_embeddings,
    multilabel_to_primary_aspect,
    split_dev_three_way,
    tune_calib_thresholds,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOK_DIR = os.path.join(ROOT, 'models', 'phobert_multitask')


def test_split_dev_three_way_sizes_disjoint_and_rare_strata():
    df_dev = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'dev_clean.csv'))
    assert len(df_dev) >= 1100

    eye, cal, blk = split_dev_three_way(df_dev, random_state=42)
    assert (len(eye), len(cal), len(blk)) == (250, 300, len(df_dev) - 550)

    s_eye, s_cal, s_blk = set(eye.index), set(cal.index), set(blk.index)
    assert len(s_eye & s_cal) == 0
    assert len(s_eye & s_blk) == 0
    assert len(s_cal & s_blk) == 0
    assert len(s_eye | s_cal | s_blk) == len(df_dev)

    # Repeatability with fixed random_state
    eye2, cal2, blk2 = split_dev_three_way(df_dev, random_state=42)
    assert list(eye.index) == list(eye2.index)
    assert list(cal.index) == list(cal2.index)
    assert list(blk.index) == list(blk2.index)

    # Edge case: rare (sentiment, aspect_group) strata with count 1 and 2 must not crash
    rng = np.random.default_rng(42)
    sents = rng.choice([0, 1, 2], size=1112, p=[0.32, 0.20, 0.48])
    asps = rng.choice([0, 1, 2, 3], size=1112)
    sents[0], asps[0] = 1, 4  # singleton stratum
    sents[1], asps[1] = 2, 4  # size-2 stratum
    sents[2], asps[2] = 2, 4
    df_rare = pd.DataFrame({'sentiment': sents, 'aspect_group': asps})
    e_r, c_r, b_r = split_dev_three_way(df_rare, random_state=42)
    assert (len(e_r), len(c_r), len(b_r)) == (250, 300, 562)
    assert len(set(e_r.index) | set(c_r.index) | set(b_r.index)) == 1112


def test_phobert_v3_architecture_semantic_init_and_dynamic_collate():
    assert MAX_LEN == 256
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    cfg = AutoConfig.from_pretrained(TOK_DIR)
    model = PhoBertMultiTask(config=cfg, dropout=0.3)

    tok_ids = init_semantic_pii_embeddings(model.encoder, tok)
    assert set(tok_ids.keys()) == set(PII_TOKENS)
    assert len(tok) == 64006
    assert tok.mask_token_id == 64000
    assert model.encoder.get_input_embeddings().weight.shape[0] == len(tok)

    # Every new PII token must equal the mean of its Vietnamese semantic phrase embeddings
    emb_w = model.encoder.get_input_embeddings().weight.detach()
    for pii_tok, phrase in PII_SEED_PHRASES.items():
        ref_ids = tok.encode(phrase, add_special_tokens=False)
        assert len(ref_ids) >= 1 and tok.unk_token_id not in ref_ids
        expected_vec = emb_w[ref_ids].mean(dim=0)
        assert torch.allclose(emb_w[tok.convert_tokens_to_ids(pii_tok)], expected_vec, atol=1e-5)

    # Tokenizer preserves PII special tokens as single pieces
    pieces = tok.tokenize('Liên hệ [PHONE] mã [ORDER_ID] qua [EMAIL]')
    assert '[PHONE]' in pieces and '[ORDER_ID]' in pieces and '[EMAIL]' in pieces

    # ViSFDDataset + dynamic_collate_fn pad dynamically to batch max (<= 256)
    ds = ViSFDDataset(
        ['pin trâu', 'màn_hình rất đẹp nha bạn'],
        tok,
        sents=[2, 2],
        aspects=[[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]],
        max_len=MAX_LEN,
    )
    batch = dynamic_collate_fn([ds[0], ds[1]], pad_id=tok.pad_token_id)
    assert batch['input_ids'].shape[0] == 2
    assert batch['input_ids'].shape[1] < MAX_LEN
    assert batch['attention_mask'].shape == batch['input_ids'].shape
    assert batch['sentiment'].dtype == torch.long and batch['sentiment'].shape == (2,)
    assert batch['aspects'].dtype == torch.float32 and batch['aspects'].shape == (2, 5)

    # Truncation cap at MAX_LEN = 256
    long_batch = dynamic_collate_fn([
        {'input_ids': list(range(300)), 'attention_mask': [1] * 300, 'sentiment': 0, 'aspects': [1, 0, 0, 0, 0]},
        {'input_ids': [0, 5, 2], 'attention_mask': [1, 1, 1], 'sentiment': 2, 'aspects': [0, 0, 0, 0, 1]},
    ], pad_id=tok.pad_token_id)
    assert long_batch['input_ids'].shape == (2, MAX_LEN)
    assert long_batch['input_ids'][1, 3:].eq(tok.pad_token_id).all()
    assert long_batch['attention_mask'][1, 3:].eq(0).all()

    ls, la = model(batch['input_ids'], batch['attention_mask'])
    assert ls.shape == (2, 3) and la.shape == (2, 5)


def test_temperature_scaler_calibration_and_primary_fallback():
    rng = np.random.default_rng(42)
    y_sent = rng.integers(0, 3, size=300)
    y_aspects = (rng.random((300, 5)) > 0.65).astype(int)
    y_aspects[y_aspects.sum(axis=1) == 0, 4] = 1

    # Deliberately overconfident logits (high magnitude with ~20% confident errors) so T > 1 lowers NLL/BCE
    sent_logits = rng.normal(0, 0.5, size=(300, 3))
    for i, ys in enumerate(y_sent):
        if rng.random() < 0.8:
            sent_logits[i, ys] += 4.5
        else:
            sent_logits[i, (ys + 1) % 3] += 4.5
    flip_mask = rng.random((300, 5)) < 0.2
    wrong_asp = 1 - y_aspects
    eff_asp = np.where(flip_mask, wrong_asp, y_aspects)
    asp_logits = np.where(eff_asp == 1, 4.5, -4.5) + rng.normal(0, 0.4, size=(300, 5))

    scaler = TemperatureScaler().fit(sent_logits, asp_logits, y_sent, y_aspects)
    assert scaler.temp_sent > 1.0
    assert scaler.temp_asp > 1.0

    # Verify post-calibration NLL and BCE strictly improve over uncalibrated T=1.0
    uncal = TemperatureScaler(1.0, 1.0)
    sp0, ap0 = uncal.predict_probs(sent_logits, asp_logits)
    sp, ap = scaler.predict_probs(sent_logits, asp_logits)
    nll_before = -np.log(np.clip(sp0[np.arange(300), y_sent], 1e-12, 1.0)).mean()
    nll_after = -np.log(np.clip(sp[np.arange(300), y_sent], 1e-12, 1.0)).mean()
    bce_before = -(y_aspects * np.log(np.clip(ap0, 1e-12, 1.0)) + (1 - y_aspects) * np.log(np.clip(1 - ap0, 1e-12, 1.0))).mean()
    bce_after = -(y_aspects * np.log(np.clip(ap, 1e-12, 1.0)) + (1 - y_aspects) * np.log(np.clip(1 - ap, 1e-12, 1.0))).mean()
    assert nll_after < nll_before
    assert bce_after < bce_before
    assert sp.shape == (300, 3) and ap.shape == (300, 5)
    assert np.allclose(sp.sum(axis=-1), 1.0, atol=1e-6)
    assert (ap >= 0.0).all() and (ap <= 1.0).all()

    restored = TemperatureScaler.from_dict(scaler.to_dict())
    assert restored.temp_sent == scaler.temp_sent and restored.temp_asp == scaler.temp_asp

    th = tune_calib_thresholds(sp, ap, y_sent, y_aspects)
    for key in ['aspect_threshold', 'low_conf_threshold', 'calib_aspect_macro_f1', 'calib_review_precision', 'calib_review_fpr']:
        assert key in th
    assert 0.25 <= th['aspect_threshold'] <= 0.75
    assert 0.30 <= th['low_conf_threshold'] <= 0.80

    # Primary aspect fallback uses argmax when all probabilities are < threshold
    low_probs = np.array([[0.10, 0.35, 0.20, 0.05, 0.15], [0.10, 0.80, 0.60, 0.05, 0.05]])
    assert list(multilabel_to_primary_aspect(low_probs, threshold=0.5)) == [1, 1]


def test_notebook04_v3_imports_syntax_and_early_stopping():
    nb_path = os.path.join(ROOT, 'notebooks', '04_phobert_multitask.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = ["".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for idx, src in enumerate(code_cells):
        ast.parse(src, filename=f'cell_{idx}')

    full_src = "\n".join(code_cells)
    assert 'parse_label_colab' not in full_src
    assert 'clean_and_seg_colab' not in full_src
    assert 'training_logs = [' not in full_src
    assert 'hp_candidates' not in full_src
    assert 'src.modeling' in full_src
    assert 'src.labels' in full_src
    assert 'src.text_cleaner' in full_src
    assert 'split_dev_three_way' in full_src
    assert 'init_semantic_pii_embeddings' in full_src
    assert 'dynamic_collate_fn' in full_src
    assert 'TemperatureScaler' in full_src
    assert 'tune_calib_thresholds' in full_src
    assert 'BCEWithLogitsLoss' in full_src
    assert 'CrossEntropyLoss' in full_src
    assert 'PATIENCE = 2' in full_src and 'break' in full_src
    assert 'dev_pred_phobert.csv' in full_src and 'test_pred_phobert.csv' in full_src
    assert 'calibration.json' in full_src
    assert 'phobert_ecommerce_outputs_v3.zip' in full_src
