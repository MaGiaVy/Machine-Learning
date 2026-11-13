import json
import os
import re
import sys
import unittest

import pandas as pd
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.text_cleaner import check_no_content, clean_text, segment_words


class TestTask1TextCleaningAndFilter(unittest.TestCase):
    def setUp(self):
        self.root = ROOT_DIR
        self.nb00_path = os.path.join(self.root, 'notebooks', '00_load_and_merge_raw.ipynb')
        self.nb01_path = os.path.join(self.root, 'notebooks', '01_text_cleaning_and_pii.ipynb')
        self.nb02_path = os.path.join(self.root, 'notebooks', '02_eda.ipynb')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')
        self.fig_dir = os.path.join(self.root, 'reports', 'figures')

    def test_notebooks_use_src_package_and_executed(self):
        """Kiểm tra NB 00, 01, 02 dùng trực tiếp package src/ và đã thực thi toàn bộ cell."""
        for p in [self.nb00_path, self.nb01_path, self.nb02_path]:
            self.assertTrue(os.path.exists(p), f"Không tìm thấy notebook {p}")
            with open(p, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
            self.assertTrue(all(c.get('execution_count') is not None for c in code_cells), f"Notebook chưa chạy hết cell: {p}")

        with open(self.nb00_path, 'r', encoding='utf-8') as f:
            code00 = "\n".join("".join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code')
        self.assertIn("from src.labels import", code00)
        self.assertNotIn("def resolve_sentiment", code00, "NB 00 không được chứa hàm gán nhãn phụ thuộc n_star")

        with open(self.nb01_path, 'r', encoding='utf-8') as f:
            code01 = "\n".join("".join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code')
        self.assertIn("from src.text_cleaner import", code01)
        self.assertIn("lowercase=False", code01)
        self.assertNotIn("def clean_text", code01, "NB 01 phải dùng src.text_cleaner thay vì nhân bản code")
        self.assertIn("drop_duplicates", code01)

        with open(self.nb02_path, 'r', encoding='utf-8') as f:
            code02 = "\n".join("".join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code')
        self.assertIn("AutoTokenizer", code02)
        self.assertIn("fmt='.1f'", code02)

    def test_cleaning_logic_on_sample_phrases(self):
        """Kiểm tra src.text_cleaner trên câu mẫu: bảo tồn hoa/thường, nguyên âm, tiếng Anh và lọc spam."""
        self.assertEqual(clean_text("máy dùng quáaaa đãaa"), "máy dùng quá đã")
        self.assertEqual(clean_text("máy dùng đẹp quáaaa"), "máy dùng đẹp quá")
        self.assertEqual(clean_text("rấttt tốttt"), "rất tốt")
        self.assertEqual(
            clean_text("TGDĐ freee full app call kkk lưu ý"),
            "Thế Giới Di Động free full app call cười lưu ý",
        )
        seg_pii = segment_words("Gọi [PHONE] đơn [ORDER_ID]")
        self.assertIn("[PHONE]", seg_pii)
        self.assertIn("[ORDER_ID]", seg_pii)

        self.assertFalse(check_no_content("Máy dùng tốt"))
        self.assertFalse(check_no_content("Sản phẩm tuyệt vời"))
        self.assertTrue(check_no_content("ok"))
        self.assertTrue(check_no_content("nhận xu"))
        self.assertTrue(check_no_content("cho 5 sao nhận xu"))
        self.assertTrue(check_no_content(""))
        self.assertTrue(check_no_content("   "))

    def test_clean_datasets_content_integrity(self):
        """Kiểm tra train_clean (7776), dev_clean (1110), test_clean (2224) giữ chữ hoa/thường và không OOV."""
        splits = {
            'train': (os.path.join(self.proc_dir, 'train_clean.csv'), 7776),
            'dev': (os.path.join(self.proc_dir, 'dev_clean.csv'), 1110),
            'test': (os.path.join(self.proc_dir, 'test_clean.csv'), 2224),
        }

        for name, (path, expected_len) in splits.items():
            self.assertTrue(os.path.exists(path), f"Không tìm thấy file {path}")
            df = pd.read_csv(path)
            self.assertEqual(len(df), expected_len, f"Tập {name} phải có đúng {expected_len} dòng")

            pos_mask = df['sentence'].str.contains('tốt|tuyệt vời', case=False, na=False)
            no_content_pos = df[pos_mask & (df['is_no_content'] == True)]
            self.assertEqual(len(no_content_pos), 0)

            oov_cnt = df['sentence_clean'].str.contains(r'\b(?:quáa|đãa)\b', case=False, na=False).sum()
            self.assertEqual(oov_cnt, 0, f"Tập {name} còn chứa {oov_cnt} từ lỗi quáa/đãa")

            cased_cnt = df['sentence_clean'].fillna('').str.contains(r'[A-ZÀ-Ỹ]', regex=True).sum()
            self.assertGreater(cased_cnt, 0, f"Tập {name} bị mất chữ viết hoa (cased) cho PhoBERT")

    def test_label_parsing_no_extreme_negative_bias(self):
        """Kiểm tra phân phối nhãn thuần Majority-over-3 tại Notebook 00 không bị ép theo n_star."""
        df_tr = pd.read_csv(os.path.join(self.proc_dir, 'train_raw.csv'))
        star5 = df_tr[df_tr['n_star'] == 5]
        neg_ratio_5star = (star5['sentiment'] == 0).mean()
        self.assertLess(neg_ratio_5star, 0.08, f"5-star negative ratio too high: {neg_ratio_5star:.2%}")

        star4 = df_tr[df_tr['n_star'] == 4]
        neg_ratio_4star = (star4['sentiment'] == 0).mean()
        self.assertLess(neg_ratio_4star, 0.25, f"4-star negative ratio too high: {neg_ratio_4star:.2%}")

        neu_ratio = (df_tr['sentiment'] == 1).mean()
        self.assertGreaterEqual(neu_ratio, 0.15, f"Neutral/Mixed class collapsed: {neu_ratio:.2%}")

    def test_pyvi_segmentation_and_teencode(self):
        """Kiểm tra tách từ pyvi, chuẩn hóa teencode tgdđ và bảo toàn ngoặc PII."""
        df_te = pd.read_csv(os.path.join(self.proc_dir, 'test_clean.csv'))
        has_tgdd = df_te['sentence_clean'].fillna('').str.contains(r'(?<!\w)tgdđ?(?!\w)', case=False, regex=True).any()
        self.assertFalse(has_tgdd, "Teencode 'tgdđ'/'tgdd' was not normalized")
        seg_all = " ".join(df_te['sentence_seg'].fillna(''))
        self.assertIn("nhiệt_tình", seg_all.lower())
        self.assertIn("trải_nghiệm", seg_all.lower())
        broken_pii = df_te['sentence_seg'].fillna('').str.contains(
            r'\[\s+(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)|(?:URL|EMAIL|ORDER_ID|PHONE|PERSON)\s+\]',
            regex=True,
        ).sum()
        self.assertEqual(broken_pii, 0, "pyvi inserted spaces inside PII brackets")

    def test_eda_figures_generated_and_valid(self):
        """Kiểm tra 4 biểu đồ EDA từ NB 02 tồn tại và là ảnh PNG hợp lệ."""
        figs = [
            'eda_sentiment_dist.png',
            'eda_aspect_dist.png',
            'eda_star_vs_sentiment.png',
            'eda_aspect_sentiment_heatmap.png',
        ]
        for fname in figs:
            fpath = os.path.join(self.fig_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Thiếu biểu đồ {fpath}")
            self.assertGreater(os.path.getsize(fpath), 10000, f"Biểu đồ {fname} quá nhỏ")
            with Image.open(fpath) as img:
                img.verify()
                self.assertEqual(img.format, 'PNG')


def test_v3_pure_labels_dedup_and_clean_integrity():
    tr = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'train_clean.csv'))
    de = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'dev_clean.csv'))
    te = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'test_clean.csv'))
    merged_path = os.path.join(ROOT_DIR, 'data', 'processed', 'visfd_merged_raw.csv')
    assert os.path.exists(merged_path)
    df_merged = pd.read_csv(merged_path)
    assert len(df_merged) == 11122
    assert set(df_merged['split'].unique()) == {'train', 'dev', 'test'}

    # No n_star leakage: must contain 4-5 star Negative OR 1-2 star Positive reviews in ground truth
    true_contra = ((te['n_star'] <= 2) & (te['sentiment'] == 2)) | ((te['n_star'] >= 4) & (te['sentiment'] == 0))
    assert true_contra.sum() >= 50, f"Expected >= 50 real star contradictions in Test, got {true_contra.sum()}"

    # Multi-label columns present across splits
    for df_split in (tr, de, te):
        for c in ['asp_0', 'asp_1', 'asp_2', 'asp_3', 'asp_4']:
            assert c in df_split.columns and set(df_split[c].unique()).issubset({0, 1})
        assert 'all_aspects' in df_split.columns

    # Zero clean-text leakage across Train vs Dev/Test and zero internal duplicates in Train/Dev
    s_tr = set(tr['sentence_clean'].fillna(''))
    s_de = set(de['sentence_clean'].fillna(''))
    s_te = set(te['sentence_clean'].fillna(''))
    assert len(s_tr & s_de) == 0 and len(s_tr & s_te) == 0 and len(s_de & s_te) == 0
    assert tr['sentence_clean'].duplicated().sum() == 0
    assert de['sentence_clean'].duplicated().sum() == 0

    # Check is_no_content column integrity (UIT-ViSFD has 0 empty/spam rows; verify filter on synthetic inputs)
    assert tr['is_no_content'].dtype == bool
    assert int(tr['is_no_content'].sum()) == 0
    assert check_no_content("cho 5 sao nhận xu") is True
    assert check_no_content("ok") is True

    # Ensure clean text preserved 'affordable' and 'lưu ý' without regex corruption
    assert not tr['sentence_clean'].fillna('').str.contains(r'\baf \[ORDER_ID\]\b').any()
    assert not tr['sentence_clean'].fillna('').str.contains(r'(?<!\w)(?:lư ý|hữ)(?!\w)', case=False).any()


if __name__ == '__main__':
    unittest.main()
