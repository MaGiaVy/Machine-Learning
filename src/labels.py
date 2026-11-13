import re
import unicodedata

ASPECT_GROUPS = ['PERF_FEAT', 'SCREEN_CAM', 'BATTERY', 'PRICE_SER', 'DESIGN_GEN']

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
    """Giải mã nhãn UIT-ViSFD thuần túy theo Majority-over-3 và Multi-Label 5 nhóm."""
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

    #Nhãn cảm xúc thuần từ {ASPECT#Polarity}, tuyệt đối không đọc n_star
    if not pairs:
        return {
            'sentiment': 1,
            'aspect_group': 4 if not all_aspects else all_aspects[0],
            'aspects': aspects,
            'all_aspects': all_aspects,
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
    }
