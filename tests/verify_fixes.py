"""
Script nghiệm thu toàn diện (Verification Suite) cho dự án VoC-Insight v2.0
Kiểm tra toàn bộ 6 Task theo kế hoạch D:\\E-commerce\\docs\\plans\\2026-11-06-fix-methodology-and-eyeball-review.md
"""

import os
import sys
import json
import joblib
import unittest
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, 'notebooks')
FIGURES_DIR = os.path.join(ROOT_DIR, 'reports', 'figures')

class TestComprehensiveVerification(unittest.TestCase):
    """Bộ kiểm định toàn diện chất lượng mã nguồn và dữ liệu VoC-Insight v2.0."""

    def test_01_task1_text_preprocessing_and_clean_datasets(self):
        nb01_path = os.path.join(NOTEBOOKS_DIR, '01_text_cleaning_and_pii.ipynb')
        self.assertTrue(os.path.exists(nb01_path), f"Không tìm thấy {nb01_path}")
        with open(nb01_path, 'r', encoding='utf-8') as f:
            nb01 = json.load(f)
        code01 = "\n".join("".join(c['source']) for c in nb01['cells'] if c['cell_type'] == 'code')
        
        self.assertIn(r"re.sub(r'([a-zÀ-ỹ])\1{2,}', r'\1', t)", code01)
        self.assertNotIn(r"\1\1", code01)
        self.assertIn("ViTokenizer.tokenize", code01)

        splits = {
            'train': (os.path.join(DATA_DIR, 'train_clean.csv'), 7786),
            'dev': (os.path.join(DATA_DIR, 'dev_clean.csv'), 1112),
            'test': (os.path.join(DATA_DIR, 'test_clean.csv'), 2224)
        }

        for split_name, (file_path, expected_rows) in splits.items():
            self.assertTrue(os.path.exists(file_path), f"File {file_path} không tồn tại")
            df = pd.read_csv(file_path)
            self.assertEqual(len(df), expected_rows)
            mask_positive = df['sentence'].str.contains('tốt|tuyệt vời', case=False, na=False)
            false_no_content = df[mask_positive & (df['is_no_content'] == True)]
            self.assertEqual(len(false_no_content), 0)
            oov_count = df['sentence_clean'].str.contains(r'\b(?:quáa|đãa)\b', case=False, na=False).sum()
            self.assertEqual(oov_count, 0)

    def test_02_task2_baseline_eyeball_blackbox_and_models(self):
        nb03_path = os.path.join(NOTEBOOKS_DIR, '03_baseline_tfidf_models.ipynb')
        self.assertTrue(os.path.exists(nb03_path), f"Không tìm thấy {nb03_path}")
        with open(nb03_path, 'r', encoding='utf-8') as f:
            nb03 = json.load(f)
        code03 = "\n".join("".join(c['source']) for c in nb03['cells'] if c['cell_type'] == 'code')
        
        self.assertNotIn("from sklearn.pipeline import Pipeline", code03)
        self.assertIn("df_eyeball", code03)
        self.assertIn("df_blackbox", code03)
        self.assertIn("f1_train", code03)

        vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
        svc_s_path = os.path.join(MODELS_DIR, 'svc_sentiment.pkl')
        svc_a_path = os.path.join(MODELS_DIR, 'svc_aspect.pkl')
        
        for p in [vec_path, svc_s_path, svc_a_path]:
            self.assertTrue(os.path.exists(p), f"Không tìm thấy file model {p}")

        vec = joblib.load(vec_path)
        svc_s = joblib.load(svc_s_path)
        svc_a = joblib.load(svc_a_path)

        self.assertIsInstance(vec, TfidfVectorizer)
        self.assertTrue(hasattr(vec, 'vocabulary_'))
        self.assertIsInstance(svc_s, LinearSVC)
        self.assertIsInstance(svc_a, LinearSVC)

        base_pred_path = os.path.join(DATA_DIR, 'test_pred_baseline.csv')
        self.assertTrue(os.path.exists(base_pred_path))
        df_base = pd.read_csv(base_pred_path)
        self.assertEqual(len(df_base), 2224)
        sent_f1 = f1_score(df_base['sentiment'], df_base['pred_sentiment'], average='macro')
        asp_f1 = f1_score(df_base['aspect_group'], df_base['pred_aspect'], average='macro')
        self.assertGreater(sent_f1, 0.60)
        self.assertGreater(asp_f1, 0.58)

    def test_03_task3_phobert_multitask_and_metadata(self):
        nb04_path = os.path.join(NOTEBOOKS_DIR, '04_phobert_multitask.ipynb')
        self.assertTrue(os.path.exists(nb04_path))
        with open(nb04_path, 'r', encoding='utf-8') as f:
            nb04 = json.load(f)
        code04 = "\n".join("".join(c['source']) for c in nb04['cells'] if c['cell_type'] == 'code')

        self.assertIn("0.9 * loss_asp_fn", code04)
        self.assertNotIn("1.2 * loss_asp_fn", code04)
        self.assertNotIn("df_test['sentiment'].tolist()", code04)
        self.assertNotIn("training_logs = [", code04)
        self.assertIn("raise FileNotFoundError", code04)

        meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        df_base = pd.read_csv(os.path.join(DATA_DIR, 'test_pred_baseline.csv'))
        df_pho = pd.read_csv(os.path.join(DATA_DIR, 'test_pred_phobert.csv'))

        base_meta = meta["baseline_tfidf_linearsvc"]
        self.assertEqual(base_meta["sentiment_macro_f1"], round(f1_score(df_base['sentiment'], df_base['pred_sentiment'], average='macro'), 4))
        self.assertEqual(base_meta["aspect_macro_f1"], round(f1_score(df_base['aspect_group'], df_base['pred_aspect'], average='macro'), 4))

        phobert_meta = meta["phobert_multitask"]
        self.assertEqual(phobert_meta["backbone"], "vinai/phobert-base-v2")
        self.assertEqual(phobert_meta["sentiment_macro_f1"], round(f1_score(df_pho['sentiment'], df_pho['pred_sentiment'], average='macro'), 4))
        self.assertEqual(phobert_meta["aspect_macro_f1"], round(f1_score(df_pho['aspect_group'], df_pho['pred_aspect'], average='macro'), 4))

    def test_04_task4_actionable_insights_and_shap_figure(self):
        nb05_path = os.path.join(NOTEBOOKS_DIR, '05_shap_and_actionable_insights.ipynb')
        self.assertTrue(os.path.exists(nb05_path))
        with open(nb05_path, 'r', encoding='utf-8') as f:
            nb05_src = f.read()
        self.assertNotIn("else df_neg", nb05_src)
        self.assertNotIn("-abs(x[1])", nb05_src)
        self.assertIn("sub = df_neg[df_neg['pred_aspect'] == asp_id]", nb05_src)

        insights_path = os.path.join(DATA_DIR, 'actionable_insights.csv')
        df_insights = pd.read_csv(insights_path)
        df_pho = pd.read_csv(os.path.join(DATA_DIR, 'test_pred_phobert.csv'))
        self.assertEqual(len(df_insights), 6)
        self.assertEqual(int(df_insights['negative_count'].sum()), int((df_pho['pred_sentiment'] == 0).sum()))

        shap_fig_path = os.path.join(FIGURES_DIR, 'shap_top_negative_words.png')
        self.assertTrue(os.path.exists(shap_fig_path))
        self.assertGreater(os.path.getsize(shap_fig_path), 5000)
        with Image.open(shap_fig_path) as img:
            img.verify()
            self.assertEqual(img.format, 'PNG')

    def test_notebook06_pii_masked_and_preprocessed(self):
        excel_path = os.path.join(DATA_DIR, 'sample_excel_predicted.xlsx')
        self.assertTrue(os.path.exists(excel_path), f"Không tìm thấy {excel_path}")
        df_out = pd.read_excel(excel_path)
        assert 'sentence_clean' in df_out.columns, "NB06 output missing cleaned & PII-masked text column"
        assert 'sentence_seg' in df_out.columns, "NB06 output missing pyvi word-segmented column"
        raw_phones = df_out['sentence_clean'].fillna('').str.contains(r'\b0[0-9]{9}\b').any()
        assert not raw_phones, "Unmasked phone numbers leaked into sample_excel_predicted.xlsx"


def test_notebook06_pii_masked_and_preprocessed():
    df_out = pd.read_excel(os.path.join(DATA_DIR, 'sample_excel_predicted.xlsx'))
    assert 'sentence_clean' in df_out.columns, "NB06 output missing cleaned & PII-masked text column"
    raw_phones = df_out['sentence_clean'].fillna('').str.contains(r'\b0[0-9]{9}\b').any()
    assert not raw_phones, "Unmasked phone numbers leaked into sample_excel_predicted.xlsx"

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], '-v'])
