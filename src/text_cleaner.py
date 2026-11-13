from itertools import product
import re
import unicodedata

#Bảng ánh xạ emoji cảm xúc sang từ tiếng Việt gạch dưới chuẩn PhoBERT
_EMOJI_MAP = {
    '❤️': 'yêu_thích', '❤': 'yêu_thích', '♥': 'yêu_thích', '😍': 'yêu_thích',
    '🥰': 'yêu_thích', '😘': 'yêu_thích', '😙': 'yêu_thích', '😚': 'yêu_thích',
    '🤩': 'yêu_thích', '💖': 'yêu_thích', '💗': 'yêu_thích', '💓': 'yêu_thích',
    '💕': 'yêu_thích', '💞': 'yêu_thích', '💘': 'yêu_thích', '🤗': 'yêu_thích',
    '🫶': 'yêu_thích', '💙': 'yêu_thích', '💚': 'yêu_thích', '💛': 'yêu_thích',
    '💜': 'yêu_thích', '🧡': 'yêu_thích', '🖤': 'yêu_thích', '🤍': 'yêu_thích',
    '🤎': 'yêu_thích',
    '👍': 'hài_lòng', '👌': 'hài_lòng', '💯': 'hài_lòng', '✨': 'hài_lòng',
    '😊': 'hài_lòng', '😁': 'hài_lòng', '😄': 'hài_lòng', '😀': 'hài_lòng',
    '😃': 'hài_lòng', '😉': 'hài_lòng', '🙂': 'hài_lòng', '☺': 'hài_lòng',
    '😌': 'hài_lòng', '👏': 'hài_lòng', '🙌': 'hài_lòng', '🎉': 'hài_lòng',
    '🔥': 'hài_lòng', '🥳': 'hài_lòng', '🙆': 'hài_lòng', '💪': 'hài_lòng',
    '😂': 'cười', '🤣': 'cười', '😆': 'cười', '😜': 'cười',
    '🤪': 'cười', '😝': 'cười', '😛': 'cười', '😋': 'cười',
    '😡': 'tức_giận', '😠': 'tức_giận', '🤬': 'tức_giận', '💢': 'tức_giận',
    '👿': 'tức_giận',
    '😭': 'buồn_bã', '😢': 'buồn_bã', '😥': 'buồn_bã', '😪': 'buồn_bã',
    '😿': 'buồn_bã', '😰': 'buồn_bã', '😓': 'buồn_bã', '😨': 'buồn_bã',
    '😱': 'buồn_bã',
    '😞': 'thất_vọng', '😔': 'thất_vọng', '😒': 'thất_vọng', '😑': 'thất_vọng',
    '😐': 'thất_vọng', '😩': 'thất_vọng', '😫': 'thất_vọng', '💔': 'thất_vọng',
    '🙄': 'thất_vọng', '🤦': 'thất_vọng', '🤮': 'thất_vọng', '🤢': 'thất_vọng',
    '🤕': 'thất_vọng', '🤒': 'thất_vọng', '🤧': 'thất_vọng', '😵': 'thất_vọng',
    '🥴': 'thất_vọng', '😬': 'thất_vọng', '😕': 'thất_vọng', '😟': 'thất_vọng',
    '🙁': 'thất_vọng', '☹': 'thất_vọng',
    '👎': 'không_hài_lòng', '💩': 'không_hài_lòng', '💀': 'không_hài_lòng',
    '☠': 'không_hài_lòng',
    '⭐': 'sao', '☆': 'sao', '🌟': 'sao',
    '🔋': 'pin', '🐃': 'trâu', '📸': 'camera', '📷': 'camera',
    '📱': 'điện_thoại', '🎧': 'tai_nghe', '🎮': 'game',
}

_EMOJI_KEYS_RE = re.compile(
    '(' + '|'.join(re.escape(k) for k in sorted(_EMOJI_MAP, key=len, reverse=True)) + r')(?:\s*\1)+'
)
_EMOJI_SINGLE_RE = re.compile(
    '|'.join(re.escape(k) for k in sorted(_EMOJI_MAP, key=len, reverse=True))
)

#Loại bỏ symbol/emoji không hỗ trợ để tránh sinh <unk> trong PhoBERT
_UNSUPPORTED_CHAR_RE = re.compile(r'[^\x20-\x7e\n\r\t\u00b0\u00c0-\u024f\u1e00-\u1eff\u2026]')

#Regex ẩn danh PII trước khi rút gọn ký tự hoặc tách từ
_FULL_URL_RE = re.compile(r'https?://\S+?(?=[.,;:!?)]*(?:\s|$))|www\.\S+?(?=[.,;:!?)]*(?:\s|$))', re.IGNORECASE)
_EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(
    r'(?<![\w@/.])(?:thegioididong\.com|dienmayxanh\.com|fptshop\.com\.vn|shopee\.vn|tiki\.vn|lazada\.vn|tiktok\.com|cellphones\.com\.vn|viettelstore\.vn|hoanghamobile\.com|[a-zA-Z][\w-]{1,}\.(?:com|vn|com\.vn))(?:/\S+?)?(?=[.,;:!?)]*(?:\s|$))',
    re.IGNORECASE,
)
_ORDER_PREFIX_RE = re.compile(r'(?<![A-Za-zÀ-ỹ0-9])#?(?:dh|đh|ord|spx)[-_]?[0-9][0-9a-zA-Z_-]{3,}\b', re.IGNORECASE)
_ORDER_NUM_RE = re.compile(r'(?<!\d)[1-9][0-9]{11,17}(?!\d)')
_PHONE_RE = re.compile(r'(?<!\d)(?:\+84[\s\.\-]?|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-46-9])(?:[\s\.\-]?\d){7}(?!\d)')

_SURNAMES = (
    'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan', 'Vũ', 'Võ',
    'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý', 'Đoàn', 'Trương', 'Đinh',
)
_PERSON_FULL_RE = re.compile(
    r'(?<!\w)(?:' + '|'.join(_SURNAMES) + r')(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){2,3}(?!\w)'
)
_PERSON_ADJ_LEFT_RE = re.compile(
    r'(?<!\w)((?:' + '|'.join(_SURNAMES) + r')\s+[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,2})(\s*[,:\-]?\s*(?:_PHONE_|_ORDER_ID_))'
)
_PERSON_ADJ_RIGHT_RE = re.compile(
    r'((?:_PHONE_|_ORDER_ID_)\s*[,:\-]?\s*)((?:' + '|'.join(_SURNAMES) + r')\s+[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,2})(?!\w)'
)

_NON_PERSON_PHRASES = (
    'thế giới di động', 'điện máy xanh', 'nhân viên', 'samsung', 'iphone',
    'oppo', 'xiaomi', 'shopee', 'lazada', 'tiki', 'tiktok', 'hồ chí minh',
    'quận', 'phường', 'huyện', 'tỉnh', 'thành phố', 'chi nhánh', 'cửa hàng',
    'đơn hàng', 'mã đơn', 'số điện', 'khách hàng', 'sản phẩm', 'giao hàng',
    'bảo hành', 'việt nam', 'hà nội', 'đà nẵng', 'cần thơ', 'bình thạnh',
    'gò vấp', 'thanh xuân', 'ninh kiều', 'hóc môn', 'tân bình', 'thủ đức',
)
_ADDRESS_PREFIX_RE = re.compile(
    r'(?:chi\s+nhánh|địa\s+chỉ|đ/c|đường|phố|khúc|số\s*\d+[a-z]?|cửa\s+hàng|tgdđ|tgdd|đmx|điện\s+máy\s+xanh|thế\s+giới\s+di\s+động|\b\d{1,4}[a-z]?(?:-\d{1,4})?)\s+$',
    re.IGNORECASE,
)
_ADDRESS_SUFFIX_RE = re.compile(r'^\s*,?\s*(?:q\.?\s*\d+|quận|phường|huyện|xã|hóc\s+môn|bình\s+tân|gò\s+vấp|thủ\s+đức)', re.IGNORECASE)
_PERSON_CUE_PREFIX_RE = re.compile(r'(?:tên|khách(?:\s+hàng)?|anh|chị|bạn|em)\s+$', re.IGNORECASE)

#Bảo vệ tiếng cười trước bước rút gọn ký tự
_LAUGH_K_RE = re.compile(r'(?<!\w)[kK]{3,}(?!\w)')
_LAUGH_HA_RE = re.compile(r'(?<!\w)(?:[hH][aA]){2,}[hH]?(?!\w)')

#Rút gọn nguyên âm tiếng Việt kéo dài theo họ nguyên âm (tuyệt đối tách ư và u)
_VOWEL_FAMILY_PATTERNS = [
    (re.compile(r'([àáảãạăằắẳẵặâầấẩẫậ])[aăâ]+', re.IGNORECASE), r'\1'),
    (re.compile(r'([èéẻẽẹêềếểễệ])[eê]+', re.IGNORECASE), r'\1'),
    (re.compile(r'([òóỏõọôồốổỗộơờớởỡợ])[oôơ]+', re.IGNORECASE), r'\1'),
    (re.compile(r'([ùúủũụ])u+', re.IGNORECASE), r'\1'),
    (re.compile(r'([ưừứửữự])ư+', re.IGNORECASE), r'\1'),
    (re.compile(r'([ìíỉĩị])i+', re.IGNORECASE), r'\1'),
    (re.compile(r'([ỳýỷỹỵ])y+', re.IGNORECASE), r'\1'),
]

#Whitelist từ có 2 chữ cái liền kề hợp lệ để không rút gọn nhầm về 1 chữ
_DOUBLE_LETTER_WORDS = {
    'free', 'good', 'cool', 'full', 'app', 'call', 'mess', 'pass', 'boss',
    'look', 'book', 'feel', 'need', 'keep', 'seed', 'week', 'zoom', 'room',
    'moon', 'soon', 'oppo', 'google', 'shopee', 'tgdd', 'ss', 'facebook',
    'apple', 'messenger', 'screen', 'xoong', 'boong', 'haizz', 'comment',
    'setting', 'bluetooth', 'sweet', 'speed', 'sleep', 'green', 'miss',
    'less', 'loss', 'food', 'tool', 'pool', 'door', 'floor', 'wood',
    'bill', 'will', 'sell', 'well', 'tell', 'hall', 'wall', 'mall', 'tall', 'fall',
}
_WORD_RE = re.compile(r'[A-Za-zÀ-ỹ]+')
_TRIPLE_RUN_RE = re.compile(r'([A-Za-zÀ-ỹ])\1{2,}', re.IGNORECASE)

#Từ điển teencode theo ranh giới từ an toàn
_MN_RE = re.compile(r'(?<![\w.])m\.n(?![\w.])', re.IGNORECASE)
_COLON_V_RE = re.compile(r'(?<![:/\w]):[vV](?=\s|$|[.,!?])')
_CAM_RE = re.compile(r'(?<!màu\s)(?<!\w)cam(?!\w)(?!\s+(?:ứng|kết|đoan|chịu))', re.IGNORECASE)
_SINGLE_TEENCODE = {'k': 'không', 'm': 'mình', 'v': 'vậy', 'r': 'rồi', 'j': 'gì', 'z': 'vậy'}
_SINGLE_RE = re.compile(r'(?<![\w.:/\-])([kmvrjzKMVRJZ])(?![\w.:/\-])')

_PROPER_TEENCODE = {
    'tgdd': 'Thế Giới Di Động',
    'tgdđ': 'Thế Giới Di Động',
    'tgd': 'Thế Giới Di Động',
    'đmx': 'Điện Máy Xanh',
    'dmx': 'Điện Máy Xanh',
    'fpt': 'FPT Shop',
}
_TEENCODE_DICT = {
    'ko': 'không', 'kh': 'không', 'hong': 'không', 'hok': 'không', 'hk': 'không', 'kg': 'không',
    'đc': 'được', 'dc': 'được', 'đk': 'được', 'dk': 'được',
    'sp': 'sản phẩm', 'đt': 'điện thoại', 'dt': 'điện thoại',
    'nv': 'nhân viên', 'nvtv': 'nhân viên tư vấn',
    'bh': 'bảo hành', 'ttbh': 'trung tâm bảo hành',
    'tgdd': 'thế giới di động', 'tgdđ': 'thế giới di động', 'tgd': 'thế giới di động',
    'đmx': 'điện máy xanh', 'dmx': 'điện máy xanh', 'fpt': 'fpt shop',
    'mk': 'mình', 'mik': 'mình', 'mn': 'mọi người', 'mng': 'mọi người',
    'bth': 'bình thường', 'bt': 'bình thường', 'sd': 'sử dụng',
    'trc': 'trước', 'vs': 'với', 'cx': 'cũng',
    'okie': 'ok', 'oke': 'ok', 'oki': 'ok',
    'tl': 'trả lời', 'ib': 'nhắn tin', 'đh': 'đơn hàng',
    'hqua': 'hôm qua', 'hnay': 'hôm nay', 'ntn': 'như thế nào',
    'ae': 'anh em', 'ng': 'người', 'tks': 'cảm ơn',
}
_MULTI_TEENCODE_RE = re.compile(
    r'(?<!\w)(' + '|'.join(re.escape(k) for k in sorted(_TEENCODE_DICT, key=len, reverse=True)) + r')(?!\w)',
    re.IGNORECASE,
)

_PII_RESTORE = [
    ('_URL_', '[URL]'),
    ('_EMAIL_', '[EMAIL]'),
    ('_ORDER_ID_', '[ORDER_ID]'),
    ('_PHONE_', '[PHONE]'),
    ('_PERSON_', '[PERSON]'),
]
_PII_TOKEN_RE = re.compile(r'\[(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)\]')
_PII_TOKEN_SPLIT_RE = re.compile(r'(\[(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)\])')
_PYVI_BRACKET_FIX_RE = re.compile(r'\[\s*(URL|EMAIL|ORDER_ID|PHONE|PERSON)\s*\]')

_SPAM_PHRASES = (
    'nhận xu', 'lấy xu', 'cho 5 sao', 'chưa dùng',
    'chưa thử', 'không có ý kiến', 'đã nhận hàng',
)
_ALNUM_WORD_RE = re.compile(r'[0-9A-Za-zÀ-ỹ]+')

_VI_TOKENIZER = None


def _is_valid_person_phrase(phrase: str, full_text: str, start: int, end: int) -> bool:
    low = phrase.lower()
    if any(bad in low for bad in _NON_PERSON_PHRASES):
        return False
    left_ctx = full_text[max(0, start - 25):start]
    right_ctx = full_text[end:min(len(full_text), end + 20)]
    near_pii = ('_PHONE_' in left_ctx) or ('_ORDER_ID_' in left_ctx) or ('_PHONE_' in right_ctx) or ('_ORDER_ID_' in right_ctx)
    if (_ADDRESS_PREFIX_RE.search(left_ctx) or _ADDRESS_SUFFIX_RE.search(right_ctx)) and not near_pii and not _PERSON_CUE_PREFIX_RE.search(left_ctx):
        return False
    return True


def _mask_person(text: str) -> str:
    def _repl_full(m: re.Match) -> str:
        phrase = m.group(0)
        if _is_valid_person_phrase(phrase, text, m.start(), m.end()):
            return ' _PERSON_ '
        return phrase

    out = _PERSON_FULL_RE.sub(_repl_full, text)

    def _repl_adj_left(m: re.Match) -> str:
        phrase, tail = m.group(1), m.group(2)
        if any(bad in phrase.lower() for bad in _NON_PERSON_PHRASES):
            return m.group(0)
        return ' _PERSON_ ' + tail

    def _repl_adj_right(m: re.Match) -> str:
        head, phrase = m.group(1), m.group(2)
        if any(bad in phrase.lower() for bad in _NON_PERSON_PHRASES):
            return m.group(0)
        return head + ' _PERSON_ '

    out = _PERSON_ADJ_LEFT_RE.sub(_repl_adj_left, out)
    out = _PERSON_ADJ_RIGHT_RE.sub(_repl_adj_right, out)
    return out


def _match_case(orig: str, repl: str) -> str:
    if orig.lower() in _PROPER_TEENCODE and any(c.isupper() for c in orig):
        return _PROPER_TEENCODE[orig.lower()]
    if repl == 'ok' and orig.isupper():
        return 'OK'
    if orig[0].isupper() and (len(orig) == 1 or not orig.isupper()):
        return repl[0].upper() + repl[1:]
    return repl


def _dedup_word(m: re.Match) -> str:
    word = m.group(0)
    runs = list(_TRIPLE_RUN_RE.finditer(word))
    if not runs:
        return word

    #Thử khôi phục các từ tiếng Anh/thương hiệu có cặp 2 chữ cái giống nhau như free, full, app, tgdd
    if len(runs) <= 3:
        for choices in product((2, 1), repeat=len(runs)):
            pieces = []
            prev = 0
            for run, keep in zip(runs, choices):
                pieces.append(word[prev:run.start()])
                pieces.append(run.group(1) * keep)
                prev = run.end()
            pieces.append(word[prev:])
            cand = ''.join(pieces)
            if cand.lower() in _DOUBLE_LETTER_WORDS:
                return cand

    return _TRIPLE_RUN_RE.sub(r'\1', word)


def clean_text(text: str, lowercase: bool = False) -> str:
    """Làm sạch bình luận tiếng Việt giữ chữ Hoa/Thường mặc định cho PhoBERT."""
    if text is None or (isinstance(text, float) and text != text):
        return ''

    # 1. Chuẩn hóa Unicode NFC và loại bỏ Variation Selector / ZWJ / skin-tone trước tiên
    t = unicodedata.normalize('NFC', str(text))
    t = t.replace('\ufe0f', '').replace('\ufe0e', '').replace('\u200d', '')
    t = re.sub(r'[\U0001f3fb-\U0001f3ff]', '', t)
    t = t.replace('\u2105', '%')
    t = re.sub(r'(?<=[A-Za-zÀ-ỹ])\u20ab|\u20ab(?=[A-Za-zÀ-ỹ])', 'đ', t)
    t = t.replace('\u20ab', ' đ')
    t = t.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('–', '-').replace('—', '-')

    # 2. Chuyển emoji cảm xúc sang từ tiếng Việt gạch dưới và xóa icon lạ tránh <unk>
    t = _EMOJI_KEYS_RE.sub(r'\1', t)
    t = _EMOJI_SINGLE_RE.sub(lambda m: f' {_EMOJI_MAP[m.group(0)]} ', t)
    t = _UNSUPPORTED_CHAR_RE.sub(' ', t)

    # 3. Ẩn danh PII trước khi rút gọn ký tự hay tách từ
    t = _FULL_URL_RE.sub(' _URL_ ', t)
    t = _EMAIL_RE.sub(' _EMAIL_ ', t)
    t = _BARE_DOMAIN_RE.sub(' _URL_ ', t)
    t = _ORDER_PREFIX_RE.sub(' _ORDER_ID_ ', t)
    t = _ORDER_NUM_RE.sub(' _ORDER_ID_ ', t)
    t = _PHONE_RE.sub(' _PHONE_ ', t)
    t = _mask_person(t)

    # 4. Bảo vệ tiếng cười và rút gọn ký tự kéo dài an toàn (giữ nguyên ưu/free/full/app/call)
    t = _LAUGH_K_RE.sub(' cười ', t)
    t = _LAUGH_HA_RE.sub(' cười ', t)
    for pat, repl in _VOWEL_FAMILY_PATTERNS:
        t = pat.sub(repl, t)
    t = _WORD_RE.sub(_dedup_word, t)

    # 5. Chuẩn hóa teencode theo ngữ cảnh và ranh giới từ
    t = re.sub(r'\s+', ' ', t).strip()
    t = _MN_RE.sub(lambda m: _match_case(m.group(0), 'mọi người'), t)
    t = _COLON_V_RE.sub(' haha ', t)
    t = _CAM_RE.sub(lambda m: _match_case(m.group(0), 'camera'), t)

    t_before_single = t

    def _repl_single(m: re.Match) -> str:
        ch = m.group(1)
        low = ch.lower()
        #Tránh chuyển nhầm đơn vị đo sau chữ số như '4 M' hay '5 V'
        if low in ('m', 'v', 'r', 'z') and re.search(r'\d\s+$', t_before_single[:m.start()]):
            return ch
        return _match_case(ch, _SINGLE_TEENCODE[low])

    t = _SINGLE_RE.sub(_repl_single, t)
    t_before_multi = t

    def _repl_multi(m: re.Match) -> str:
        tok = m.group(1)
        low = tok.lower()
        if low == 'kg' and re.search(r'\d\s+$', t_before_multi[:m.start()]):
            return tok
        return _match_case(tok, _TEENCODE_DICT[low])

    t = _MULTI_TEENCODE_RE.sub(_repl_multi, t)

    # 6. Khôi phục placeholder PII chuẩn và xử lý lowercase nếu được yêu cầu
    for tmp_tok, real_tok in _PII_RESTORE:
        t = t.replace(tmp_tok, real_tok)
    t = re.sub(r'\s+', ' ', t).strip()

    if lowercase:
        parts = _PII_TOKEN_SPLIT_RE.split(t)
        t = ''.join(p if i % 2 == 1 else p.lower() for i, p in enumerate(parts))

    return t


def check_no_content(cleaned_text: str) -> bool:
    """Trả về True nếu câu rỗng, dưới 2 từ có nghĩa (trừ PII), hoặc câu ngắn săn xu."""
    if not isinstance(cleaned_text, str) or not cleaned_text.strip():
        return True
    no_pii = _PII_TOKEN_RE.sub(' ', cleaned_text)
    words = _ALNUM_WORD_RE.findall(no_pii)
    if len(words) < 2:
        return True
    if len(words) <= 8:
        low = no_pii.lower()
        if any(p in low for p in _SPAM_PHRASES):
            return True
    return False


def segment_words(cleaned_text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi và khôi phục nguyên vẹn các placeholder PII."""
    global _VI_TOKENIZER
    if not isinstance(cleaned_text, str) or not cleaned_text.strip():
        return ''
    if _VI_TOKENIZER is None:
        from pyvi import ViTokenizer
        _VI_TOKENIZER = ViTokenizer
    tok = _VI_TOKENIZER.tokenize(cleaned_text)
    tok = re.sub(r'\b([Tt])rải nghiệm\b', r'\1rải_nghiệm', tok)
    tok = _PYVI_BRACKET_FIX_RE.sub(r'[\1]', tok)
    return tok
