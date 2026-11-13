import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModel

MAX_LEN = 256
PII_TOKENS = ['[PHONE]', '[ORDER_ID]', '[EMAIL]', '[URL]', '[PERSON]']
PII_SEED_PHRASES = {
    '[PHONE]': 'số điện_thoại',
    '[ORDER_ID]': 'mã đơn_hàng',
    '[EMAIL]': 'email',
    '[URL]': 'đường_dẫn',
    '[PERSON]': 'khách_hàng',
}


def _build_safe_strata(df: pd.DataFrame, min_count: int) -> pd.Series | None:
    sent = df['sentiment'].astype(str)
    if 'aspect_group' not in df.columns:
        vc = sent.value_counts()
        return sent if (vc >= 2).all() and len(vc) > 1 else None

    comb = sent + '_' + df['aspect_group'].astype(str)
    vc = comb.value_counts()
    rare = vc[vc < min_count].index
    strat = comb.where(~comb.isin(rare), sent)

    # Collapse any lone fallback sentiment stratum (< 2 samples) into sentiment-only
    vc2 = strat.value_counts()
    too_small = vc2[vc2 < 2].index
    if len(too_small) > 0:
        bad_sents = set(sent[strat.isin(too_small)])
        strat = strat.where(~sent.isin(bad_sents), sent)
        vc3 = strat.value_counts()
        if (vc3 < 2).any() or len(vc3) < 2:
            vc_s = sent.value_counts()
            return sent if (vc_s >= 2).all() and len(vc_s) > 1 else None
    return strat


def split_dev_three_way(
    df_dev: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split Dev (~1,112 rows) into disjoint Eyeball (250), Calib (300), and Blackbox (rest/562)."""
    strat1 = _build_safe_strata(df_dev, min_count=3)
    df_eyeball, df_rest = train_test_split(
        df_dev, train_size=250, stratify=strat1, random_state=random_state
    )
    strat2 = _build_safe_strata(df_rest, min_count=2)
    df_calib, df_blackbox = train_test_split(
        df_rest, train_size=300, stratify=strat2, random_state=random_state
    )
    return df_eyeball, df_calib, df_blackbox


class ViSFDDataset(Dataset):
    """Tokenizes segmented text without padding so dynamic_collate_fn pads per batch."""

    def __init__(self, texts, tokenizer, sents=None, aspects=None, max_len: int = MAX_LEN):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.sents = None if sents is None else list(sents)
        self.aspects = None if aspects is None else np.asarray(aspects, dtype=np.float32)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }
        if self.sents is not None:
            item['sentiment'] = torch.tensor(int(self.sents[idx]), dtype=torch.long)
        if self.aspects is not None:
            item['aspects'] = torch.tensor(self.aspects[idx], dtype=torch.float32)
        return item


class PhoBertMultiTask(nn.Module):
    """Shared PhoBERT encoder with 3-class Sentiment head and 5-class Multi-Label Aspect head."""

    def __init__(self, model_name_or_dir: str = 'vinai/phobert-base-v2', config=None, dropout: float = 0.3):
        super().__init__()
        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_dir)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden, 3)
        self.asp_head = nn.Linear(hidden, 5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = self.drop(out.last_hidden_state[:, 0, :])
        return self.sent_head(cls_vec), self.asp_head(cls_vec)


def init_semantic_pii_embeddings(encoder: nn.Module, tokenizer) -> dict[str, int]:
    """Register PII special tokens and initialize each embedding from its Vietnamese semantic phrase."""
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in PII_TOKENS if tok not in vocab]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)

    emb_layer = encoder.get_input_embeddings()
    if emb_layer.weight.shape[0] < len(tokenizer):
        encoder.resize_token_embeddings(len(tokenizer))
        emb_layer = encoder.get_input_embeddings()

    tok_ids = {}
    unk_id = tokenizer.unk_token_id
    with torch.no_grad():
        weight = emb_layer.weight
        for tok, phrase in PII_SEED_PHRASES.items():
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            ref_ids = [i for i in tokenizer.encode(phrase, add_special_tokens=False) if i != unk_id]
            if not ref_ids:
                ref_ids = tokenizer.encode(phrase, add_special_tokens=False)
            if tok_id is not None and tok_id != unk_id and ref_ids:
                weight[tok_id].copy_(weight[ref_ids].mean(dim=0))
                tok_ids[tok] = tok_id
    return tok_ids


def dynamic_collate_fn(batch: list[dict], pad_id: int = 1) -> dict[str, torch.Tensor]:
    """Pad batch sequences dynamically up to MAX_LEN (256) and stack sentiment/aspect targets."""
    max_len = min(MAX_LEN, max(torch.as_tensor(item['input_ids']).numel() for item in batch))
    ids_list, mask_list = [], []
    for item in batch:
        ids = torch.as_tensor(item['input_ids'], dtype=torch.long).reshape(-1)[:max_len]
        mask = torch.as_tensor(item['attention_mask'], dtype=torch.long).reshape(-1)[:max_len]
        pad_n = max_len - ids.shape[0]
        if pad_n > 0:
            ids = torch.cat([ids, torch.full((pad_n,), pad_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_n, dtype=torch.long)])
        ids_list.append(ids)
        mask_list.append(mask)

    out = {
        'input_ids': torch.stack(ids_list),
        'attention_mask': torch.stack(mask_list),
    }
    if 'sentiment' in batch[0]:
        out['sentiment'] = torch.stack([torch.as_tensor(item['sentiment'], dtype=torch.long) for item in batch])
    if 'aspects' in batch[0]:
        out['aspects'] = torch.stack([torch.as_tensor(item['aspects'], dtype=torch.float32) for item in batch])
    return out


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0))),
        np.exp(np.clip(logits, -80.0, 80.0)) / (1.0 + np.exp(np.clip(logits, -80.0, 80.0))),
    )


class TemperatureScaler:
    """Post-hoc temperature scaling fitted on df_calib logits for sentiment and multi-label aspects."""

    def __init__(self, temp_sent: float = 1.0, temp_asp: float = 1.0):
        self.temp_sent = float(temp_sent)
        self.temp_asp = float(temp_asp)

    def fit(self, sent_logits, asp_logits, y_sent, y_aspects) -> 'TemperatureScaler':
        z_s = _to_numpy(sent_logits).astype(np.float64)
        y_s = _to_numpy(y_sent).astype(np.int64)
        z_a = _to_numpy(asp_logits).astype(np.float64)
        y_a = _to_numpy(y_aspects).astype(np.float64)

        def _nll_sent(t_arr):
            scaled = z_s / t_arr[0]
            shifted = scaled - scaled.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            return float(-log_probs[np.arange(len(y_s)), y_s].mean())

        def _bce_asp(t_arr):
            x = z_a / t_arr[0]
            return float(np.mean(np.maximum(x, 0.0) - x * y_a + np.log1p(np.exp(-np.abs(x)))))

        res_s = minimize(_nll_sent, x0=[1.0], method='L-BFGS-B', bounds=[(0.05, 10.0)])
        res_a = minimize(_bce_asp, x0=[1.0], method='L-BFGS-B', bounds=[(0.05, 10.0)])
        self.temp_sent = round(float(res_s.x[0]), 6)
        self.temp_asp = round(float(res_a.x[0]), 6)
        return self

    def predict_probs(self, sent_logits, asp_logits) -> tuple[np.ndarray, np.ndarray]:
        z_s = _to_numpy(sent_logits).astype(np.float64) / self.temp_sent
        z_a = _to_numpy(asp_logits).astype(np.float64) / self.temp_asp
        return _softmax(z_s), _sigmoid(z_a)

    def to_dict(self) -> dict[str, float]:
        return {'temp_sent': self.temp_sent, 'temp_asp': self.temp_asp}

    @classmethod
    def from_dict(cls, data: dict) -> 'TemperatureScaler':
        return cls(temp_sent=data.get('temp_sent', 1.0), temp_asp=data.get('temp_asp', 1.0))


def tune_calib_thresholds(sent_probs, asp_probs, y_sent, y_aspects) -> dict[str, float]:
    """Tune aspect_threshold for multi-label macro F1 and low_conf_threshold balancing precision/FPR."""
    sp = _to_numpy(sent_probs)
    ap = _to_numpy(asp_probs)
    ys = _to_numpy(y_sent).astype(int)
    ya = _to_numpy(y_aspects).astype(int)

    best_asp_th, best_asp_f1 = 0.5, -1.0
    for th in np.round(np.arange(0.30, 0.71, 0.02), 2):
        f1 = f1_score(ya, (ap >= th).astype(int), average='macro', zero_division=0)
        if f1 > best_asp_f1 + 1e-9 or (abs(f1 - best_asp_f1) <= 1e-9 and abs(th - 0.5) < abs(best_asp_th - 0.5)):
            best_asp_th, best_asp_f1 = float(th), float(f1)

    sent_pred = sp.argmax(axis=-1)
    asp_pred = (ap >= best_asp_th).astype(int)
    is_error = (sent_pred != ys) | (asp_pred != ya).any(axis=-1)
    sent_conf = sp.max(axis=-1)
    asp_conf = ap.max(axis=-1)

    best_conf_th, best_score, best_prec, best_fpr = 0.55, -1e9, 0.0, 0.0
    n_ok = max(int((~is_error).sum()), 1)
    for th in np.round(np.arange(0.35, 0.76, 0.02), 2):
        flagged = (sent_conf < th) | (asp_conf < th)
        tp = int((flagged & is_error).sum())
        fp = int((flagged & ~is_error).sum())
        prec = tp / max(int(flagged.sum()), 1)
        fpr = fp / n_ok
        f1_err = f1_score(is_error, flagged, zero_division=0) if is_error.any() else 0.0
        score = f1_err - (0.5 * (fpr - 0.35) if fpr > 0.35 else 0.0) - 0.1 * fpr
        if score > best_score:
            best_conf_th, best_score, best_prec, best_fpr = float(th), float(score), float(prec), float(fpr)

    return {
        'aspect_threshold': round(best_asp_th, 4),
        'low_conf_threshold': round(best_conf_th, 4),
        'calib_aspect_macro_f1': round(best_asp_f1, 4),
        'calib_review_precision': round(best_prec, 4),
        'calib_review_fpr': round(best_fpr, 4),
    }


def multilabel_to_primary_aspect(asp_probs, threshold: float = 0.5) -> np.ndarray:
    """Return highest-scoring active aspect >= threshold, falling back to argmax when all < threshold."""
    ap = _to_numpy(asp_probs)
    active = ap >= threshold
    masked = np.where(active, ap, -1.0)
    return np.where(active.any(axis=-1), masked.argmax(axis=-1), ap.argmax(axis=-1))
