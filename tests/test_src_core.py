import os
import re
import sys
import tempfile
import unicodedata

os.environ['TMP'] = os.environ['TEMP'] = 'D:/E-commerce/.tmp_run'
os.makedirs('D:/E-commerce/.tmp_run', exist_ok=True)
tempfile.tempdir = 'D:/E-commerce/.tmp_run'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.labels import ASPECT_GROUPS, parse_visfd_labels
from src.text_cleaner import clean_text, check_no_content, segment_words


def test_pure_sentiment_no_star_leakage_and_multilabel():
    assert ASPECT_GROUPS == ['PERF_FEAT', 'SCREEN_CAM', 'BATTERY', 'PRICE_SER', 'DESIGN_GEN']

    # Pure majority-over-3 without n_star; Pin kém + Tổng quan miễn chê -> both BATTERY(2) and DESIGN_GEN(4) active
    res = parse_visfd_labels("{BATTERY#Negative};{DESIGN#Positive};{GENERAL#Positive};", "Pin kém còn lại miễn chê")
    assert res["sentiment"] == 2
    assert res["aspects"] == [0, 0, 1, 0, 1]
    assert res["all_aspects"] == [2, 4]
    assert res["aspect_group"] == 2  # BATTERY found at index 0 via word-boundary keyword 'pin'

    # Boundary check: 'đánh giá' must NOT trigger PRICE_SER(3) and 'úp load' must NOT trigger PERF_FEAT(0)
    res2 = parse_visfd_labels("{PRICE#Neutral};{PERFORMANCE#Neutral};", "đánh giá sơ bộ úp load hình pin")
    assert res2["sentiment"] == 1
    assert res2["aspects"] == [1, 0, 0, 1, 0]
    assert res2["all_aspects"] == [0, 3]
    assert res2["aspect_group"] == 3  # Tie-break order [3, 2, 1, 0, 4] since neither keyword matched

    # 'cấu hình' must NOT trigger SCREEN_CAM(1) via 'hình', 'cảm giác' must NOT trigger PRICE_SER(3) via 'giá'
    res3 = parse_visfd_labels("{SCREEN#Positive};{PRICE#Positive};{PERFORMANCE#Positive};", "cảm giác cấu hình ổn, giá rẻ, màn đẹp")
    assert res3["sentiment"] == 2
    assert res3["aspects"] == [1, 1, 0, 1, 0]
    assert res3["aspect_group"] == 0  # 'cấu hình' (PERF_FEAT) occurs before 'giá' and 'màn'

    # {OTHERS} only -> Neutral (1) and DESIGN_GEN (4)
    res4 = parse_visfd_labels("{OTHERS};", "Không có ý kiến")
    assert res4 == {'sentiment': 1, 'aspect_group': 4, 'aspects': [0, 0, 0, 0, 1], 'all_aspects': [4]}

    # Tie polarity -> Neutral (1), NFD input still matches keywords after NFC normalization
    nfd_comment = unicodedata.normalize('NFD', "Thiết kế đẹp nhưng pin kém")
    res5 = parse_visfd_labels("{DESIGN#Positive};{BATTERY#Negative};", nfd_comment)
    assert res5["sentiment"] == 1
    assert res5["aspect_group"] == 2  # Prefer negative aspect in mixed sentiment


def test_text_cleaner_and_pyvi_placeholder_protection():
    raw = (
        "TGDĐ đt quáaaa đẹppp, freee full app call, lưu ý kkk màu cam "
        "affordable ORD-2024-001 sđt 090-123-4567 tại thegioididong.com chụp.net ❤️ anh nhân viên"
    )
    cleaned = clean_text(raw, lowercase=False)
    assert "Thế Giới Di Động" in cleaned
    assert "điện thoại" in cleaned
    assert "quá" in cleaned and "quáa" not in cleaned
    assert "đẹp" in cleaned and "đẹpp" not in cleaned
    assert "free full app call" in cleaned
    assert "lưu ý" in cleaned and "lư ý" not in cleaned
    assert "cười" in cleaned and "không" not in cleaned
    assert "màu cam" in cleaned and "màu camera" not in cleaned
    assert "affordable" in cleaned and "af [ORDER_ID]" not in cleaned
    assert "[ORDER_ID]" in cleaned and "[PHONE]" in cleaned and "[URL]" in cleaned
    assert "chụp.net" in cleaned and "anh nhân viên" in cleaned
    assert "yêu_thích" in cleaned

    seg = segment_words(cleaned)
    for tok in ["[ORDER_ID]", "[PHONE]", "[URL]"]:
        assert tok in seg and f"[ {tok[1:-1]} ]" not in seg

    assert check_no_content("cho 5 sao nhận xu") is True
    assert check_no_content("Máy dùng tốt") is False


def test_pii_emojis_dedup_and_teencode_edge_cases():
    # Phone variants, email, URL paths, order IDs, capitalized Vietnamese names
    txt = (
        "Khách hàng Phạm Thị Thanh Hải đặt SPX_12345 và #ĐH889911 hoặc 24051912345678, "
        "liên hệ +84 912 345 678 / 090.123.4567 hoặc user.test@gmail.com qua shopee.vn/item 😡👎😭⭐"
    )
    cleaned = clean_text(txt, lowercase=False)
    assert "[PERSON]" in cleaned and "Phạm Thị Thanh Hải" not in cleaned
    assert cleaned.count("[ORDER_ID]") == 3
    assert cleaned.count("[PHONE]") == 2
    assert "[EMAIL]" in cleaned and "[URL]" in cleaned
    assert "tức_giận" in cleaned and "không_hài_lòng" in cleaned and "buồn_bã" in cleaned and "sao" in cleaned

    # Lowercase mode must still preserve uppercase special placeholders
    cleaned_low = clean_text("Nguyễn Văn An sđt 0901234567", lowercase=True)
    assert "[PERSON]" in cleaned_low and "[PHONE]" in cleaned_low

    # Never mask generic lowercase phrases, brands, or store names
    safe_txt = "anh nhân viên và anh em dùng tên sản phẩm tại Thế Giới Di Động, Điện Máy Xanh, Samsung, iPhone"
    assert "[PERSON]" not in clean_text(safe_txt)

    # Double-letter English/brand whitelist & uu preservation
    dbl_txt = "goood coool messs passs shopeee oppooo tgddd TGDDD ss cứuuu hữu ích sưu tầm"
    dbl_clean = clean_text(dbl_txt)
    for expected in ["good", "cool", "mess", "pass", "shopee", "oppo", "thế giới di động", "Thế Giới Di Động", "ss", "cứu", "hữu ích", "sưu tầm"]:
        assert expected in dbl_clean, f"Missing '{expected}' in '{dbl_clean}'"

    # Non-shouting teencode expansion on uppercase abbreviations & blue heart emoji
    assert clean_text("NV TGDĐ lễ phép 💙") == "nhân viên Thế Giới Di Động lễ phép yêu_thích"

    # Boundary-safe teencode: m.n, :v, contextual cam, single letters not broken on . or :
    teen_txt = "M.n ơi cam sau 8 M pin ổn k? máy cam ứng hơi lag :v k. là viết tắt"
    teen_clean = clean_text(teen_txt)
    assert "Mọi người" in teen_clean
    assert "camera sau" in teen_clean
    assert "cam ứng" in teen_clean
    assert "8 M" in teen_clean and "mình" not in teen_clean
    assert "không?" in teen_clean
    assert "haha" in teen_clean
    assert "k." in teen_clean

    # check_no_content edge cases
    assert check_no_content("") is True
    assert check_no_content("[PHONE] [ORDER_ID]") is True
    assert check_no_content("ok") is True
    assert check_no_content("lấy xu") is True
    assert check_no_content("chưa dùng") is True
    assert check_no_content("chưa thử") is True
    assert check_no_content("không có ý kiến") is True
    assert check_no_content("đã nhận hàng") is True
    assert check_no_content("Sản phẩm dùng rất ổn định, cho 5 sao nhận xu luôn nhé mọi người") is False

    # Pyvi segment_words protects all 5 PII tokens
    all_pii = "Khách [PERSON] đơn [ORDER_ID] gọi [PHONE] mail [EMAIL] tại [URL]"
    seg_pii = segment_words(all_pii)
    for tok in ["[PERSON]", "[ORDER_ID]", "[PHONE]", "[EMAIL]", "[URL]"]:
        assert tok in seg_pii


def test_real_uit_visfd_train_integrity():
    df_tr = pd.read_csv('data/raw/Train.csv')
    assert len(df_tr) == 7786

    # Check pure Majority-over-3 label distribution (~19.57% Neutral)
    parsed = [parse_visfd_labels(r['label'], r['comment']) for _, r in df_tr.iterrows()]
    neu_ratio = sum(p['sentiment'] == 1 for p in parsed) / len(parsed)
    assert 0.18 <= neu_ratio <= 0.21, f"Unexpected Neutral ratio: {neu_ratio:.4f}"

    # Check clean_text across ALL 7,786 Train rows
    cleaned = [clean_text(c, lowercase=False) for c in df_tr['comment']]

    assert not any(re.search(r'af\s*\[ORDER_ID\]', c) for c in cleaned)
    assert not any(re.search(r'(?<!\w)(?:lư ý|hữ)(?!\w)', c, re.IGNORECASE) for c in cleaned)
    # Clean must never truncate 'full/app/call' into 'ful/ap/cal' or turn 'màu cam' into 'màu camera'
    raw_nfc = [unicodedata.normalize('NFC', str(x)) for x in df_tr['comment']]
    assert not any(re.search(r'(?<!\w)ful(?!\w)', c, re.IGNORECASE) for c in cleaned)
    assert sum(1 for c in cleaned if re.search(r'(?<!\w)(?:ful|ap|cal)(?!\w)', c, re.IGNORECASE)) <= sum(
        1 for r in raw_nfc if re.search(r'(?<!\w)(?:ful|ap|cal)(?!\w)', r, re.IGNORECASE)
    )
    assert sum(1 for c in cleaned if re.search(r'màu camera', c, re.IGNORECASE)) <= sum(
        1 for r in raw_nfc if re.search(r'màu camera', r, re.IGNORECASE)
    )
    for r, c in zip(raw_nfc, cleaned):
        if re.search(r'màu cam(?!\w)', r, re.IGNORECASE):
            assert not re.search(r'màu camera', c, re.IGNORECASE)

    # Ensure every row containing PII placeholders preserves exact brackets after segment_words
    pii_rows = [t for t in cleaned if re.search(r'\[(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)\]', t)]
    assert len(pii_rows) > 0
    for t in pii_rows:
        seg = segment_words(t)
        assert not re.search(r'\[\s+(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)|(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)\s+\]', seg)


def test_clean_text_skip_backward_compatible():
    sample = "TGDĐ đt đẹppp quáaaa cam sau m k dùng đc sđt 0901234567 ❤️"
    assert clean_text(sample) == clean_text(sample, skip=frozenset())
    assert "yêu_thích" not in clean_text(sample, skip=frozenset({'emoji'}))
    assert "0901234567" in clean_text(sample, skip=frozenset({'pii'}))
    assert "đẹppp" in clean_text(sample, skip=frozenset({'elongation'}))
    assert "camera" not in clean_text(sample, skip=frozenset({'cam'}))
    assert "mình" not in clean_text(sample, skip=frozenset({'single_teencode'}))
    assert "Thế Giới Di Động" not in clean_text(sample, skip=frozenset({'multi_teencode'}))
