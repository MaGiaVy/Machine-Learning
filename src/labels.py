import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

ASPECT_GROUPS = ['PERF_FEAT', 'SCREEN_CAM', 'BATTERY', 'PRICE_SER', 'DESIGN_GEN']
ABSA_STATES = ['None', 'Negative', 'Neutral', 'Positive']
ABSA_STATE_VI = ['Không có', 'Tiêu cực', 'Trung lập', 'Tích cực']
ABSA_COLS = [f'absa_{i}' for i in range(5)]
_ABSA_POL_MAP = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
# Ưu tiên giữ lỗi khi cùng 1 nhóm có nhiều tag trái cực tính: Neg(1) > Neu(2) > Pos(3) > None(0)
_POL_PRIORITY = {0: 0, 3: 1, 2: 2, 1: 3}

ASPECT_MAP = {
    'PERFORMANCE': 0,
    'STORAGE': 0,
    'FEATURES': 0,
    'SCREEN': 1,
    'CAMERA': 1,
    'BATTERY': 2,
    'PRICE': 3,
    'SER&ACC': 3,
    'DESIGN': 4,
    'GENERAL': 4,
    'OTHERS': 4,
}

POLARITY_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

#Ưu tiên các nhóm cụ thể hiếm hơn khi không tìm thấy từ khóa trong câu
_TIE_BREAK_ORDER = [3, 2, 1, 0, 4]

ASPECT_KWS = {
    0: [
        'cấu hình', 'hiệu năng', 'game', 'mượt', 'lag', 'đơ', 'giật', 'ram',
        'chip', 'bộ nhớ', 'dung lượng', 'vân tay', 'wifi', 'sóng', 'loa',
        'âm thanh', 'nghe gọi', 'lỗi', 'ứng dụng', 'phần mềm', 'tính năng',
    ],
    1: [
        'màn hình', 'màn', 'hiển thị', 'cảm ứng', 'camera', 'cam', 'chụp',
        'ảnh', 'hình', 'quay', 'nét', 'mờ', 'selfie',
    ],
    2: [
        'pin', 'sạc', 'trâu', 'tuột', 'tụt', 'hao', 'nóng', 'nhanh hết',
        'dự phòng',
    ],
    3: [
        'giá', 'tiền', 'rẻ', 'đắt', 'nhân viên', 'tư vấn', 'phục vụ',
        'bảo hành', 'đổi', 'giao hàng', 'ship', 'khuyến mãi', 'quà',
        'phụ kiện', 'tai nghe', 'ốp', 'thế giới di động', 'tgdd', 'tgdđ',
        'đmx', 'shop', 'cửa hàng',
    ],
    4: [
        'thiết kế', 'đẹp', 'ngoại hình', 'mẫu mã', 'kiểu dáng', 'màu',
        'vỏ', 'lưng', 'viền', 'cầm', 'nặng', 'nhẹ', 'mỏng', 'sang',
    ],
}

#Chặn các cụm từ đồng âm khác nghĩa như 'đánh giá', 'cấu hình', 'màu cam'
_KW_OVERRIDES = {
    'giá': r'(?<!đánh\s)(?<!trị\s)giá',
    'hình': r'(?<!cấu\s)(?<!ngoại\s)(?<!tình\s)hình(?!\s+như)',
    'cam': r'(?<!màu\s)cam(?!\s+(?:kết|đoan|ứng|chịu))',
}

def _build_kw_pattern(kws: list[str]) -> re.Pattern:
    ordered = sorted(kws, key=len, reverse=True)
    parts = [_KW_OVERRIDES.get(kw, re.escape(kw)) for kw in ordered]
    return re.compile(r'(?<!\w)(?:' + '|'.join(parts) + r')(?!\w)')

_ASPECT_KW_PATTERNS = {g: _build_kw_pattern(kws) for g, kws in ASPECT_KWS.items()}
_PAIR_RE = re.compile(r'\{([A-Z&]+)#(Positive|Neutral|Negative)\}')
_TAG_RE = re.compile(r'\{([A-Z&]+)(?:#(?:Positive|Neutral|Negative))?\}')


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ''
    return str(val)


def parse_visfd_labels(label_str: str, comment: str) -> dict:
    """Giải mã nhãn UIT-ViSFD gồm bộ v3 (giữ tương thích E01-E10) và bộ True ABSA 5x4 (absa_0..4)."""
    lbl = _safe_str(label_str)
    pairs = _PAIR_RE.findall(lbl)
    raw_tags = _TAG_RE.findall(lbl)

    aspects = [0, 0, 0, 0, 0]
    for tag in raw_tags:
        if tag in ASPECT_MAP:
            aspects[ASPECT_MAP[tag]] = 1
    if not any(aspects):
        aspects[4] = 1
    all_aspects = [idx for idx, active in enumerate(aspects) if active == 1]

    # True ABSA 5x4: bỏ {OTHERS} khỏi DESIGN_GEN; xung đột cùng nhóm giải quyết theo Neg > Neu > Pos
    absa = [0, 0, 0, 0, 0]
    grp_pols = [set(), set(), set(), set(), set()]
    for asp, pol in pairs:
        if asp in ASPECT_MAP and asp != 'OTHERS':
            g = ASPECT_MAP[asp]
            s = _ABSA_POL_MAP[pol]
            grp_pols[g].add(s)
            if _POL_PRIORITY[s] > _POL_PRIORITY[absa[g]]:
                absa[g] = s
    conflict_groups = [g for g, ps in enumerate(grp_pols) if len(ps) > 1]
    has_conflict = len(conflict_groups) > 0
    conflict_with_neg = any(1 in grp_pols[g] for g in conflict_groups)

    #Nhãn cảm xúc thuần từ {ASPECT#Polarity}, tuyệt đối không đọc n_star
    if not pairs:
        return {
            'sentiment': 1,
            'aspect_group': 4 if not all_aspects else all_aspects[0],
            'aspects': aspects,
            'all_aspects': all_aspects,
            'absa': absa,
            'has_conflict': has_conflict,
            'conflict_with_neg': conflict_with_neg,
            'conflict_groups': conflict_groups,
        }

    sents = [POLARITY_MAP[pol] for _, pol in pairs]
    neg, neu, pos = sents.count(0), sents.count(1), sents.count(2)
    if neg > pos and neg > neu:
        sent = 0
    elif pos > neg and pos > neu:
        sent = 2
    else:
        sent = 1

    #Loại bỏ GENERAL/OTHERS khi câu đã có khía cạnh cụ thể
    typed_pairs = [(asp, s) for (asp, _), s in zip(pairs, sents) if asp in ASPECT_MAP]
    spec_pairs = [(asp, s) for asp, s in typed_pairs if asp not in ('GENERAL', 'OTHERS')]
    if not spec_pairs:
        spec_pairs = typed_pairs

    #Ưu tiên khía cạnh tiêu cực khi cảm xúc tổng thể là Tiêu cực hoặc Hỗn hợp
    if 0 in sents:
        neg_cands = [(asp, s) for asp, s in spec_pairs if s == 0]
        cand_pairs = neg_cands if neg_cands else spec_pairs
    elif sent == 2:
        pos_cands = [(asp, s) for asp, s in spec_pairs if s == 2]
        cand_pairs = pos_cands if pos_cands else spec_pairs
    else:
        cand_pairs = spec_pairs

    groups = sorted(
        {ASPECT_MAP[asp] for asp, _ in cand_pairs},
        key=lambda g: _TIE_BREAK_ORDER.index(g),
    )

    #Chuẩn hóa NFC trước khi tìm từ khóa xuất hiện sớm nhất theo ranh giới từ
    text = unicodedata.normalize('NFC', _safe_str(comment)).lower()

    def _first_pos(group_id: int) -> int:
        m = _ASPECT_KW_PATTERNS[group_id].search(text)
        return m.start() if m else 10**6

    primary_aspect = min(groups, key=_first_pos) if groups else 4
    return {
        'sentiment': sent,
        'aspect_group': primary_aspect,
        'aspects': aspects,
        'all_aspects': all_aspects,
        'absa': absa,
        'has_conflict': has_conflict,
        'conflict_with_neg': conflict_with_neg,
        'conflict_groups': conflict_groups,
    }


def derive_overall_sentiment(absa_mat) -> np.ndarray:
    """Luật cảnh báo hậu xử lý từ ma trận ABSA 5x4 (Neg > Neu > Pos > None); không dùng thay majority v3."""
    mat = np.asarray(absa_mat, dtype=int)
    has_neg = (mat == 1).any(axis=-1)
    has_neu = (mat == 2).any(axis=-1)
    has_pos = (mat == 3).any(axis=-1)
    return np.where(has_neg, 0, np.where(has_neu, 1, np.where(has_pos, 2, 1)))


def compute_absa_metrics(y_true_5x4, y_pred_5x4) -> dict:
    """Tính đầy đủ bộ metric ABSA 5x4: E2E Micro/Macro, 4-class Macro, Polarity-given-detected và Baselines."""
    yt = np.asarray(y_true_5x4, dtype=int)
    yp = np.asarray(y_pred_5x4, dtype=int)
    n, k = yt.shape

    tp_all = fp_all = fn_all = 0
    aspect_e2e_f1, per_4cls_f1, maj_4cls_f1, all1_pres_f1 = [], [], [], []
    for j in range(k):
        yt_j, yp_j = yt[:, j], yp[:, j]
        tp = int(((yt_j == yp_j) & (yt_j > 0)).sum())
        fp = int(((yp_j > 0) & (yt_j != yp_j)).sum())
        fn = int(((yt_j > 0) & (yt_j != yp_j)).sum())
        tp_all += tp
        fp_all += fp
        fn_all += fn
        aspect_e2e_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        per_4cls_f1.append(float(f1_score(yt_j, yp_j, labels=[0, 1, 2, 3], average='macro', zero_division=0)))
        maj_cls = int(np.bincount(yt_j, minlength=4).argmax())
        maj_4cls_f1.append(float(f1_score(yt_j, np.full(n, maj_cls), labels=[0, 1, 2, 3], average='macro', zero_division=0)))
        all1_pres_f1.append(float(f1_score((yt_j > 0).astype(int), np.ones(n, dtype=int), zero_division=0)))

    det_mask = (yt > 0) & (yp > 0)
    pol_det_f1 = (
        float(f1_score(yt[det_mask], yp[det_mask], labels=[1, 2, 3], average='macro', zero_division=0))
        if det_mask.any() else 0.0
    )
    return {
        'absa_micro_f1': round(2 * tp_all / max(2 * tp_all + fp_all + fn_all, 1), 4),
        'absa_macro_f1': round(float(np.mean(aspect_e2e_f1)), 4),
        'mean_4cls_macro_f1': round(float(np.mean(per_4cls_f1)), 4),
        'per_aspect_4cls_f1': [round(x, 4) for x in per_4cls_f1],
        'polarity_given_detected_macro_f1': round(pol_det_f1, 4),
        'majority_4cls_macro_f1': round(float(np.mean(maj_4cls_f1)), 4),
        'all1_presence_macro_f1': round(float(np.mean(all1_pres_f1)), 4),
    }


def compute_paired_bootstrap_ci(
    y_true_5x4,
    y_pred_a,
    y_pred_b,
    metric_key: str = 'absa_macro_f1',
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI: resample 1 vector chỉ số dòng dùng chung cho cả model A và B để tính CI của hiệu số."""
    yt = np.asarray(y_true_5x4, dtype=int)
    ya = np.asarray(y_pred_a, dtype=int)
    yb = np.asarray(y_pred_b, dtype=int)
    n = len(yt)
    rng = np.random.default_rng(seed)

    vals_a, vals_b, diffs = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma = compute_absa_metrics(yt[idx], ya[idx])[metric_key]
        mb = compute_absa_metrics(yt[idx], yb[idx])[metric_key]
        vals_a[b], vals_b[b], diffs[b] = ma, mb, ma - mb

    return {
        'metric': metric_key,
        'n_boot': int(n_boot),
        'a_mean': round(float(vals_a.mean()), 4),
        'a_ci95': [round(float(np.quantile(vals_a, 0.025)), 4), round(float(np.quantile(vals_a, 0.975)), 4)],
        'b_mean': round(float(vals_b.mean()), 4),
        'b_ci95': [round(float(np.quantile(vals_b, 0.025)), 4), round(float(np.quantile(vals_b, 0.975)), 4)],
        'diff_mean': round(float(diffs.mean()), 4),
        'diff_ci95': [round(float(np.quantile(diffs, 0.025)), 4), round(float(np.quantile(diffs, 0.975)), 4)],
    }


def _strip_punct_words(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', _safe_str(text).lower()).split()


def dedup_remove_train_neardup(
    df_train: pd.DataFrame,
    ref_dfs: dict[str, pd.DataFrame],
    threshold: float = 0.90,
    min_words: int = 5,
    keep_ids: set | None = None,
    apply: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Lọc câu train gần trùng với dev/test sau khi bỏ dấu câu/chấm lặp; yêu cầu >= min_words và hỗ trợ gate 2 bước."""
    keep_set = set() if keep_ids is None else set(keep_ids)
    tr_col = 'sentence_clean' if 'sentence_clean' in df_train.columns else 'sentence'
    tr_words = [_strip_punct_words(s) for s in df_train[tr_col]]
    tr_norm = [' '.join(w) for w in tr_words]
    tr_elig = np.array([len(w) >= min_words for w in tr_words], dtype=bool)
    tr_ids = df_train['index'].tolist() if 'index' in df_train.columns else df_train.index.tolist()

    ref_texts, ref_splits, ref_raw_texts = [], [], []
    for sp_name, df_r in ref_dfs.items():
        r_col = 'sentence_clean' if 'sentence_clean' in df_r.columns else 'sentence'
        for s in df_r[r_col]:
            w = _strip_punct_words(s)
            if len(w) >= min_words:
                ref_texts.append(' '.join(w))
                ref_splits.append(sp_name)
                ref_raw_texts.append(_safe_str(s))

    max_cos = np.zeros(len(df_train), dtype=np.float32)
    best_ref = np.full(len(df_train), -1, dtype=int)
    if tr_elig.any() and ref_texts:
        tr_elig_texts = [tr_norm[i] for i in np.where(tr_elig)[0]]
        min_df = 2 if len(tr_elig_texts) >= 50 else 1
        try:
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=min_df)
            m_tr = vec.fit_transform(tr_elig_texts)
        except ValueError:
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
            m_tr = vec.fit_transform(tr_elig_texts)
        m_ref = vec.transform(ref_texts)
        elig_idx = np.where(tr_elig)[0]
        for start in range(0, m_tr.shape[0], 1000):
            sims = (m_tr[start:start + 1000] @ m_ref.T).toarray()
            rows_idx = elig_idx[start:start + 1000]
            max_cos[rows_idx] = sims.max(axis=1)
            best_ref[rows_idx] = sims.argmax(axis=1)

    drop_rows = []
    for i in np.where(max_cos >= threshold)[0]:
        rj = int(best_ref[i])
        tid = int(tr_ids[i])
        drop_rows.append({
            'train_idx': tid,
            'row_pos': int(i),
            'max_cosine': round(float(max_cos[i]), 4),
            'matched_split': ref_splits[rj],
            'train_text': _safe_str(df_train.iloc[i][tr_col]),
            'matched_ref_text': ref_raw_texts[rj],
            'approved': tid not in keep_set,
        })
    df_dropped = pd.DataFrame(drop_rows)
    if not apply or df_dropped.empty:
        return df_train.copy(), df_dropped, max_cos

    drop_pos = set(df_dropped.loc[df_dropped['approved'], 'row_pos'].tolist())
    keep_mask = [i not in drop_pos for i in range(len(df_train))]
    return df_train.loc[keep_mask].reset_index(drop=True), df_dropped, max_cos


def classify_silent_complaint_levels(stars, absa_mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính So_Khia_Canh_Tieu_Cuc (0..5), Muc_Do_Khieu_Nai_Ngam ('N/A'|'Không'|'Nhẹ'|'Cao') và Khieu_Nai_Ngam_4_5_Sao."""
    mat = np.asarray(absa_mat, dtype=int)
    neg_cnt = (mat == 1).sum(axis=1)
    pos_cnt = (mat == 3).sum(axis=1)
    n = len(mat)
    if stars is None:
        return neg_cnt, np.array(['N/A'] * n, dtype=object), np.zeros(n, dtype=bool)

    s_arr = pd.to_numeric(pd.Series(stars), errors='coerce').to_numpy()
    levels = []
    for st, nc, pc in zip(s_arr, neg_cnt, pos_cnt):
        if np.isnan(st) or st < 4:
            levels.append('N/A')
        elif nc == 0:
            levels.append('Không')
        elif nc == 1 and pc >= 1:
            levels.append('Nhẹ')
        else:
            levels.append('Cao')
    lvl_arr = np.array(levels, dtype=object)
    return neg_cnt, lvl_arr, (lvl_arr == 'Cao')


def review_exact_match(y_true, y_pred) -> float:
    """Tỉ lệ review mà cả 5 khía cạnh đều dự đoán đúng. Đầu vào (N, 5)."""
    return float((np.asarray(y_true) == np.asarray(y_pred)).all(axis=1).mean())
