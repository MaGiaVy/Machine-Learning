"""Universal dirty Excel/CSV ingestion, schema detection, and tabular PII masking."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable

import chardet
import numpy as np
import openpyxl
import pandas as pd

KW_HEADER = [
    'đánh giá', 'nhận xét', 'review', 'comment', 'nội dung', 'sao', 'rating',
    'khách', 'buyer', 'phản hồi', 'ý kiến', 'stt', 'mã', 'ưu điểm',
    'nhược điểm', 'nhóm', 'ghi chú', 'order', 'customer', 'star', 'stars',
    'sđt', 'phone', 'sentiment',
]

KW_GOOD = [
    'đánh giá', 'nhận xét', 'review', 'comment', 'nội dung', 'phản hồi',
    'ý kiến', 'ưu điểm', 'nhược điểm', 'sentiment', 'feedback', 'opinion',
    'nhan_xet', 'nhan xet', 'danh gia', 'phan hoi', 'y kien', 'noi dung',
]

KW_BAD_TOKENS = {
    'id', 'ma', 'mã', 'sdt', 'sđt', 'phone', 'sao', 'star', 'stars',
    'rating', 'name', 'ten', 'tên', 'username', 'khach', 'khách',
    'ngay', 'ngày', 'date', 'time', 'timestamp', 'stt',
    'nhom', 'nhóm', 'khu', 'vực', 'vuc',
}

_UNCONDITIONAL_BAD_TOKENS = KW_BAD_TOKENS - {'khach', 'khách'}
_IDENTITY_ONLY_TOKENS = {'khach', 'khách', 'buyer', 'customer', 'nguoi', 'người'}

_STAR_EXCLUDE_PHRASES = (
    'tại sao', 'tai sao', 'ưu điểm', 'uu diem',
    'nhược điểm', 'nhuoc diem', 'đặc điểm', 'dac diem',
)
_STAR_TOKENS = {'sao', 'star', 'stars', 'rating', 'ratings', 'điểm', 'diem', 'score'}
_STAR_SKIP_NUMERIC_TOKENS = {'stt', 'id', 'ma', 'mã', 'sdt', 'sđt', 'phone', 'tuổi', 'tuoi', 'age'}


def _norm_text(val: Any) -> str:
    return unicodedata.normalize('NFC', str(val)).lower().strip()


def _tokenize_col(col: Any) -> set[str]:
    c_low = _norm_text(col)
    return {tok for tok in re.split(r'[\s_\-\(\)\/]+', c_low) if tok}


def read_xls_safe(file_path: str | os.PathLike) -> dict[str, pd.DataFrame]:
    """Read legacy binary BIFF .xls or HTML-disguised .xls safely."""
    path = os.fspath(file_path)
    with open(path, 'rb') as f:
        head = f.read(4096).lower().replace(b'\x00', b'')
    if b'<html' in head or b'<!doctype' in head or b'<table' in head:
        return {'Sheet1': pd.read_html(path)[0]}
    return pd.read_excel(path, sheet_name=None, engine='xlrd')


def read_csv_safe(file_path: str | os.PathLike) -> dict[str, pd.DataFrame]:
    """Detect encoding with chardet (first 16KB) and delimiter (\t, ;, ,) for any encoding."""
    path = os.fspath(file_path)
    with open(path, 'rb') as f:
        raw = f.read(16384)

    if raw.startswith(b'\xef\xbb\xbf'):
        enc = 'utf-8-sig'
    elif raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        enc = 'utf-16'
    else:
        det = chardet.detect(raw)
        enc = det.get('encoding') or 'utf-8'
        if enc.lower() == 'ascii':
            enc = 'utf-8'

    txt = raw.decode(enc, errors='ignore')
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    sample = next(
        (ln for ln in lines[:10] if any(ln.count(d) >= 1 for d in ('\t', ';', ','))),
        lines[0] if lines else txt,
    )
    counts = {d: sample.count(d) for d in ('\t', ';', ',')}
    if max(counts.values()) == 0:
        counts = {d: txt.count(d) for d in ('\t', ';', ',')}
    sep = max(('\t', ';', ','), key=lambda d: counts[d]) if max(counts.values()) > 0 else ','

    return {'Sheet1': pd.read_csv(path, sep=sep, encoding=enc)}


def read_any_file(file_path: str | os.PathLike) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read .xlsx, .xls, or .csv/.tsv and return (sheets_dict, metadata)."""
    path = os.fspath(file_path)
    ext = os.path.splitext(path)[1].lower()
    meta: dict[str, Any] = {
        'file_path': path,
        'ext': ext,
        'merged_ranges': {},
        'header_loaded': True,
    }

    if ext == '.xlsx':
        meta['format'] = 'xlsx'
        meta['header_loaded'] = False
        sheets: dict[str, pd.DataFrame] = {}
        wb = openpyxl.load_workbook(path, data_only=True)
        try:
            for name in wb.sheetnames:
                ws = wb[name]
                rows = list(ws.iter_rows(values_only=True))
                sheets[name] = pd.DataFrame(rows) if rows else pd.DataFrame()
                meta['merged_ranges'][name] = [
                    (rng.min_row, rng.max_row, rng.min_col, rng.max_col)
                    for rng in ws.merged_cells.ranges
                ]
        finally:
            wb.close()
        return sheets, meta

    if ext == '.xls':
        with open(path, 'rb') as f:
            head = f.read(4096).lower().replace(b'\x00', b'')
        is_html = b'<html' in head or b'<!doctype' in head or b'<table' in head
        meta['format'] = 'html_xls' if is_html else 'biff_xls'
        return read_xls_safe(path), meta

    meta['format'] = 'csv'
    return read_csv_safe(path), meta


def pick_best_sheet(sheets_dict: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Select sheet with highest count of cells longer than 15 characters."""
    if not sheets_dict:
        raise ValueError("No sheets found in input workbook")
    best_name, best_score = None, -1
    for name, df_s in sheets_dict.items():
        if df_s.empty:
            score = 0
        else:
            score = sum(
                int((df_s[col].dropna().astype(str).str.len() > 15).sum())
                for col in df_s.columns
            )
        if score > best_score:
            best_name, best_score = name, score
    return best_name, sheets_dict[best_name]


def _count_header_kw(values: list[Any]) -> int:
    cnt = 0
    for val in values:
        if pd.isna(val):
            continue
        s = _norm_text(val)
        if not s or len(s) > 50:
            continue
        if any(k in s for k in KW_HEADER):
            cnt += 1
    return cnt


def detect_header_row(raw_df: pd.DataFrame, max_scan: int = 15) -> int:
    """Detect row index containing the e-commerce table header."""
    if raw_df.empty:
        return 0
    has_int_cols = isinstance(raw_df.columns[0], (int, np.integer))

    best_idx, best_cnt = 0, -1
    for i in range(min(max_scan, len(raw_df))):
        cnt = _count_header_kw(raw_df.iloc[i].dropna().tolist())
        if cnt > best_cnt:
            best_idx, best_cnt = i, cnt

    if not has_int_cols:
        col_names = list(raw_df.columns)
        has_unnamed = any(_norm_text(c).startswith('unnamed:') for c in col_names)
        cols_cnt = _count_header_kw(col_names)
        if not has_unnamed and cols_cnt >= 2 and cols_cnt >= best_cnt:
            return 0

    return best_idx if best_cnt >= 2 else 0


def detect_text_columns(df: pd.DataFrame) -> list[Any]:
    """Identify customer review text columns using tokenized bad-word exclusion."""
    text_cols: list[Any] = []
    for col in df.columns:
        c_low = _norm_text(col)
        col_tokens = _tokenize_col(col)
        has_good_kw = any(g in c_low for g in KW_GOOD)

        if col_tokens & _UNCONDITIONAL_BAD_TOKENS:
            continue
        if not has_good_kw and (col_tokens & _IDENTITY_ONLY_TOKENS):
            continue

        col_str = df[col].dropna().astype(str).str.strip()
        col_str = col_str[col_str != '']
        if len(col_str) == 0:
            continue

        mean_len = float(col_str.str.len().mean())
        if (has_good_kw and mean_len > 10) or mean_len > 25:
            text_cols.append(col)
    return text_cols


def detect_star_column(df: pd.DataFrame) -> Any:
    """Identify 1..5 star rating column by tokenized keyword or numeric distribution."""
    for col in df.columns:
        c_low = _norm_text(col)
        if any(ex in c_low for ex in _STAR_EXCLUDE_PHRASES):
            continue
        tokens = _tokenize_col(col)
        if tokens & _STAR_TOKENS:
            col_str = df[col].dropna().astype(str).str.strip()
            if len(col_str) == 0 or float(col_str.str.len().mean()) <= 15:
                return col

    for col in df.columns:
        tokens = _tokenize_col(col)
        if tokens & _STAR_SKIP_NUMERIC_TOKENS:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) >= 3 and (len(s) / max(len(df), 1)) >= 0.5 and float(s.between(1, 5).mean()) >= 0.8:
            return col
    return None


def _extract_header_with_rows(
    raw_df: pd.DataFrame, header_row: int | None = None
) -> tuple[pd.DataFrame, list[int], int]:
    if raw_df.empty:
        return raw_df.copy(), [], 1

    has_int_cols = isinstance(raw_df.columns[0], (int, np.integer))
    excel_row_offset = 1 if has_int_cols else 2

    if header_row is None:
        header_row = detect_header_row(raw_df)

    if header_row == 0 and not has_int_cols:
        df_c = raw_df.copy()
        excel_rows = pd.Series(np.arange(len(raw_df)) + excel_row_offset, index=df_c.index)
        header_excel_row = 1
    else:
        df_c = raw_df.iloc[header_row + 1:].copy()
        df_c.columns = [
            str(h).strip() if pd.notna(h) and str(h).strip() != '' else f'Col_{j}'
            for j, h in enumerate(raw_df.iloc[header_row])
        ]
        excel_rows = pd.Series(
            np.arange(header_row + 1, len(raw_df)) + excel_row_offset,
            index=df_c.index,
        )
        header_excel_row = header_row + excel_row_offset

    # Drop completely empty rows while preserving each surviving row's 1-based Excel row index.
    non_empty_mask = ~df_c.apply(
        lambda r: all(pd.isna(v) or str(v).strip() == '' for v in r),
        axis=1,
    )
    df_c = df_c[non_empty_mask]
    excel_rows = excel_rows[non_empty_mask]

    # Filter repeated mid-page header rows where >= 2 values match column headers.
    col_low = [_norm_text(c) for c in df_c.columns]
    keep_mask = ~df_c.apply(
        lambda r: sum(
            1 for v, c in zip(r, col_low)
            if pd.notna(v) and _norm_text(v) == c and c != ''
        ) >= 2,
        axis=1,
    )
    df_clean = df_c[keep_mask].reset_index(drop=True)
    kept_excel_rows = excel_rows[keep_mask].tolist()
    return df_clean, kept_excel_rows, header_excel_row


def extract_header_frame(raw_df: pd.DataFrame, header_row: int | None = None) -> pd.DataFrame:
    """Slice title offset rows, drop empty rows, and remove repeated mid-page header rows."""
    df_clean, _, _ = _extract_header_with_rows(raw_df, header_row=header_row)
    return df_clean


def _resolve_merged_ranges(
    meta: dict[str, Any] | None, sheet_name: str | None
) -> list[tuple[int, int, int, int]]:
    if not meta or 'merged_ranges' not in meta:
        return []
    mr_map = meta['merged_ranges']
    if sheet_name is not None:
        raw_ranges = mr_map.get(sheet_name, [])
    elif len(mr_map) == 1:
        raw_ranges = next(iter(mr_map.values()))
    else:
        raw_ranges = []

    norm_ranges: list[tuple[int, int, int, int]] = []
    for rng in raw_ranges:
        if isinstance(rng, (tuple, list)) and len(rng) == 4:
            norm_ranges.append((int(rng[0]), int(rng[1]), int(rng[2]), int(rng[3])))
        else:
            norm_ranges.append((int(rng.min_row), int(rng.max_row), int(rng.min_col), int(rng.max_col)))
    return norm_ranges


def build_clean_table(
    raw_df: pd.DataFrame,
    header_row: int | None = None,
    meta: dict[str, Any] | None = None,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Clean table, unmerge exact merged ranges only, and build unified review_text."""
    df_clean, kept_excel_rows, header_excel_row = _extract_header_with_rows(raw_df, header_row=header_row)

    merged_ranges = _resolve_merged_ranges(meta, sheet_name)
    for min_r, max_r, min_c, max_c in merged_ranges:
        if max_r <= header_excel_row:
            continue
        row_positions = [pos for pos, er in enumerate(kept_excel_rows) if min_r <= er <= max_r]
        if not row_positions:
            continue
        col_positions = [c - 1 for c in range(min_c, max_c + 1) if 0 <= c - 1 < df_clean.shape[1]]
        if not col_positions:
            continue

        top_val = df_clean.iloc[row_positions[0], col_positions[0]]
        for c_idx in col_positions:
            col_slice = df_clean.iloc[row_positions, c_idx].ffill()
            if pd.notna(top_val):
                col_slice = col_slice.fillna(top_val)
            df_clean.iloc[row_positions, c_idx] = col_slice

    txt_cols = detect_text_columns(df_clean)
    if len(txt_cols) >= 2:
        df_clean['review_text'] = df_clean[txt_cols].apply(
            lambda r: '. '.join(str(x).strip() for x in r if pd.notna(x) and str(x).strip() != ''),
            axis=1,
        )
    elif len(txt_cols) == 1:
        df_clean['review_text'] = df_clean[txt_cols[0]].fillna('').astype(str)
    else:
        df_clean['review_text'] = ''

    return df_clean


def _classify_pii_column(col: Any, text_cols: set[Any]) -> str | None:
    if col in text_cols:
        return None
    c_low = _norm_text(col)
    if any(g in c_low for g in KW_GOOD):
        return None

    tokens = _tokenize_col(col)
    c_space = re.sub(r'[\s_\-]+', ' ', c_low).strip()

    if (tokens & {'sđt', 'sdt', 'phone', 'tel', 'mobile', 'hotline'}) or ('điện thoại' in c_space) or ('dien thoai' in c_space):
        return '[PHONE]'

    if (
        c_space in {
            'mã đơn', 'ma don', 'mã đơn hàng', 'ma don hang', 'mã đh', 'ma dh',
            'order id', 'orderid', 'order code', 'order sn', 'ordersn', 'order no', 'order number',
        }
        or ('order' in tokens and bool(tokens & {'id', 'code', 'sn', 'no', 'number'}))
        or (bool(tokens & {'mã', 'ma'}) and bool(tokens & {'đơn', 'don', 'đh', 'dh'}))
    ):
        return '[ORDER_ID]'

    if (
        c_space in {
            'tên người mua', 'ten nguoi mua', 'khách hàng', 'khach hang', 'khách',
            'người mua', 'nguoi mua', 'buyer', 'buyer name', 'customer', 'customer name',
            'username', 'user name', 'họ tên', 'ho ten', 'full name', 'tên khách hàng', 'ten khach hang',
        }
        or bool(tokens & {'username', 'buyer'})
        or (bool(tokens & {'tên', 'ten', 'name'}) and not bool(tokens & {'sản', 'san', 'phẩm', 'pham', 'product', 'shop'}))
        or ('khách' in tokens or 'khach' in tokens)
    ):
        return '[PERSON]'

    if tokens & {'email', 'mail'}:
        return '[EMAIL]'

    return None


def mask_tabular_pii(
    df: pd.DataFrame, text_clean_fn: Callable[[str], str] | None = None
) -> pd.DataFrame:
    """Mask PII metadata columns and optionally clean/mask all review text columns."""
    out = df.copy()
    detected_txt = set(detect_text_columns(out))

    for col in out.columns:
        token = _classify_pii_column(col, detected_txt)
        if token is not None:
            non_empty = out[col].notna() & (out[col].astype(str).str.strip() != '')
            out[col] = out[col].astype(object)
            out.loc[non_empty, col] = token

    if text_clean_fn is not None:
        text_targets = list(detected_txt)
        for extra in ('review_text', 'combined_review', 'sentence'):
            if extra in out.columns and extra not in detected_txt:
                text_targets.append(extra)
        for col in text_targets:
            out[col] = out[col].apply(
                lambda x: text_clean_fn(str(x)) if pd.notna(x) and str(x).strip() != '' else ''
            )

    return out
