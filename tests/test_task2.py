import json
import os
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class TestTask2BaselinePipeline(unittest.TestCase):
    def setUp(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.nb_path = os.path.join(self.root, 'notebooks', '03_baseline_tfidf_models.ipynb')
        self.models_dir = os.path.join(self.root, 'models')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')

    def test_notebook03_eyeball_blackbox_and_train_eval(self):
        """Kiểm tra NB03 dùng Pipeline + StratifiedKFold thuần Train, chia Dev 3 phần và đánh giá Overfitting."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
        self.assertTrue(all(c.get('execution_count') is not None for c in code_cells), "NB03 còn cell chưa thực thi")
        for idx, cell in enumerate(code_cells):
            errors = [o for o in cell.get('outputs', []) if o.get('output_type') == 'error']
            self.assertEqual(errors, [], f"NB03 cell {idx} có lỗi thực thi: {errors}")

        all_code = "\n".join("".join(c['source']) for c in code_cells)
        nb_text = json.dumps(nb, ensure_ascii=False)

        self.assertIn('Pipeline(', all_code, "NB03 phải bọc TF-IDF + Classifier trong sklearn.pipeline.Pipeline")
        self.assertIn('StratifiedKFold', all_code, "NB03 phải dùng StratifiedKFold(5) thuần trên df_train")
        self.assertIn('OneVsRestClassifier', all_code, "NB03 phải huấn luyện Multi-Label Aspect với OneVsRestClassifier")
        self.assertIn('split_dev_three_way', all_code, "NB03 phải gọi split_dev_three_way từ src.modeling")
        self.assertIn('multilabel_to_primary_aspect', all_code, "NB03 phải gọi multilabel_to_primary_aspect")
        self.assertNotIn('PredefinedSplit', all_code, "Không được dùng PredefinedSplit làm rò rỉ Blackbox Dev")
        self.assertIn(r'\[\w+\]', all_code, "TfidfVectorizer phải giữ nguyên các PII token dạng [PHONE], [ORDER_ID]...")

        self.assertIn('df_eyeball', nb_text, "Thiếu df_eyeball trong NB03")
        self.assertIn('df_calib', nb_text, "Thiếu df_calib trong NB03")
        self.assertIn('df_blackbox', nb_text, "Thiếu df_blackbox trong NB03")
        self.assertTrue('f1_train' in nb_text or 'tr_f1' in nb_text, "Thiếu đánh giá Train F1 để kiểm tra overfitting")
        self.assertIn('f1_eyeball', nb_text, "Thiếu đánh giá Eyeball Dev F1")
        self.assertIn('f1_blackbox', nb_text, "Thiếu đánh giá Blackbox Dev F1")
        self.assertIn('smooth_w_sent', all_code, "Thiếu smooth_w_sent")
        self.assertIn('smooth_w_asp', all_code, "Thiếu smooth_w_asp")

    def test_saved_model_files_exist_and_valid(self):
        """Kiểm tra các file model baseline v3 và model tương thích được xuất đầy đủ, đúng kiểu dữ liệu."""
        expected_files = [
            'tfidf_vectorizer.pkl',
            'lr_sentiment.pkl',
            'svc_sentiment.pkl',
            'svc_aspect.pkl',
            'baseline_sentiment.pkl',
            'baseline_aspect_multilabel.pkl',
        ]
        for f in expected_files:
            p = os.path.join(self.models_dir, f)
            self.assertTrue(os.path.exists(p), f"Thiếu file model: {f}")
            obj = joblib.load(p)
            self.assertIsNotNone(obj, f"File {f} không tải được")

        vec = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
        self.assertIsInstance(vec, TfidfVectorizer)
        self.assertTrue(hasattr(vec, 'vocabulary_'), "TfidfVectorizer chưa được fit")
        for tok in ['1', 'ở', '[phone]']:
            self.assertIn(tok, vec.vocabulary_, f"TfidfVectorizer bị mất token {tok!r}")

        lr_sent = joblib.load(os.path.join(self.models_dir, 'lr_sentiment.pkl'))
        self.assertIsInstance(lr_sent, LogisticRegression)
        self.assertTrue(hasattr(lr_sent, 'coef_'), "LogisticRegression thiếu coef_ cho SHAP")

        svc_sent = joblib.load(os.path.join(self.models_dir, 'svc_sentiment.pkl'))
        self.assertIsInstance(svc_sent, LinearSVC)

        svc_asp = joblib.load(os.path.join(self.models_dir, 'svc_aspect.pkl'))
        self.assertIsInstance(svc_asp, LinearSVC)

        base_sent = joblib.load(os.path.join(self.models_dir, 'baseline_sentiment.pkl'))
        self.assertIsInstance(base_sent, Pipeline)
        self.assertIn('tfidf', base_sent.named_steps)
        self.assertIn('clf', base_sent.named_steps)

        base_asp = joblib.load(os.path.join(self.models_dir, 'baseline_aspect_multilabel.pkl'))
        self.assertIsInstance(base_asp, Pipeline)
        self.assertIsInstance(base_asp.named_steps['clf'], OneVsRestClassifier)

    def test_compatibility_with_downstream_notebooks(self):
        """Kiểm tra tương thích cả pipeline mới lẫn tfidf.transform() -> clf.predict() cho NB 05 & 06."""
        vec = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
        svc_sent = joblib.load(os.path.join(self.models_dir, 'svc_sentiment.pkl'))
        svc_asp = joblib.load(os.path.join(self.models_dir, 'svc_aspect.pkl'))
        lr_sent = joblib.load(os.path.join(self.models_dir, 'lr_sentiment.pkl'))
        base_sent = joblib.load(os.path.join(self.models_dir, 'baseline_sentiment.pkl'))
        base_asp = joblib.load(os.path.join(self.models_dir, 'baseline_aspect_multilabel.pkl'))

        sample_texts = ["điện_thoại dùng rất mượt và tốt", "pin tụt quá nhanh thất_vọng [PHONE]"]
        X_sample = vec.transform(sample_texts)

        self.assertEqual(len(svc_sent.predict(X_sample)), 2)
        self.assertEqual(len(svc_asp.predict(X_sample)), 2)
        self.assertEqual(len(lr_sent.predict(X_sample)), 2)
        self.assertEqual(len(base_sent.predict(sample_texts)), 2)
        pred_ml = base_asp.predict(sample_texts)
        self.assertEqual(pred_ml.shape, (2, 5))
        self.assertTrue(set(np.unique(pred_ml)).issubset({0, 1}))

    def test_test_pred_baseline_csv(self):
        """Kiểm tra test_pred_baseline.csv đủ 2,224 dòng, đủ cột Multi-Label Aspect và chỉ số đạt chuẩn."""
        pred_path = os.path.join(self.proc_dir, 'test_pred_baseline.csv')
        self.assertTrue(os.path.exists(pred_path), "File test_pred_baseline.csv không tồn tại")

        df = pd.read_csv(pred_path)
        self.assertEqual(len(df), 2224, "Số dòng tập test phải là 2,224")

        asp_true_cols = ['asp_0', 'asp_1', 'asp_2', 'asp_3', 'asp_4']
        asp_pred_cols = ['pred_asp_0', 'pred_asp_1', 'pred_asp_2', 'pred_asp_3', 'pred_asp_4']
        for col in ['pred_sentiment', 'pred_aspect'] + asp_pred_cols:
            self.assertIn(col, df.columns, f"Thiếu cột {col} trong test_pred_baseline.csv")
            self.assertFalse(df[col].isna().any(), f"Có giá trị NaN trong {col}")

        for col in asp_pred_cols:
            self.assertTrue(set(df[col].unique()).issubset({0, 1}), f"Cột {col} phải là nhị phân 0/1")
        self.assertTrue(set(df['pred_sentiment'].unique()).issubset({0, 1, 2}))
        self.assertTrue(set(df['pred_aspect'].unique()).issubset({0, 1, 2, 3, 4}))

        sent_f1 = f1_score(df['sentiment'], df['pred_sentiment'], average='macro')
        ml_asp_f1 = f1_score(df[asp_true_cols].values, df[asp_pred_cols].values, average='macro')
        prim_asp_f1 = f1_score(df['aspect_group'], df['pred_aspect'], average='macro')
        hit_any = float(df[asp_true_cols].values[np.arange(len(df)), df['pred_aspect'].values.astype(int)].mean())

        print(f"\n[XÁC MINH TEST SET] Baseline Sentiment Macro F1: {sent_f1:.4f}")
        print(f"[XÁC MINH TEST SET] Baseline Multi-Label Aspect Macro F1: {ml_asp_f1:.4f}")
        print(f"[XÁC MINH TEST SET] Baseline Primary Aspect Macro F1: {prim_asp_f1:.4f}")
        print(f"[XÁC MINH TEST SET] Baseline Hit-Any-True-Group: {hit_any:.4f}")

        self.assertGreater(sent_f1, 0.65, "Sentiment Macro F1 phải > 0.65")
        self.assertGreater(ml_asp_f1, 0.80, "Multi-Label Aspect Macro F1 phải > 0.80")
        self.assertGreater(prim_asp_f1, 0.45, "Primary Aspect Macro F1 phải > 0.45")
        self.assertGreater(hit_any, 0.85, "Hit-Any-True-Group phải > 0.85")


def test_v3_baseline_pipeline_no_blackbox_leak_and_multilabel():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(root, 'notebooks', '03_baseline_tfidf_models.ipynb'), encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    assert all(c.get('execution_count') is not None for c in code_cells), "NB03 has unexecuted cells"
    src_txt = "\n".join("".join(c['source']) for c in code_cells)
    assert 'StratifiedKFold' in src_txt and 'Pipeline(' in src_txt
    assert 'OneVsRestClassifier' in src_txt
    assert 'PredefinedSplit' not in src_txt, "Must not tune hyperparameters on Blackbox Dev!"
    assert 'df_eyeball' in src_txt and 'df_calib' in src_txt and 'df_blackbox' in src_txt

    base_sent = joblib.load(os.path.join(root, 'models', 'baseline_sentiment.pkl'))
    base_asp = joblib.load(os.path.join(root, 'models', 'baseline_aspect_multilabel.pkl'))
    assert isinstance(base_sent, Pipeline)
    assert isinstance(base_asp, Pipeline) and isinstance(base_asp.named_steps['clf'], OneVsRestClassifier)

    df_base = pd.read_csv(os.path.join(root, 'data', 'processed', 'test_pred_baseline.csv'))
    assert len(df_base) == 2224
    for c in ['pred_asp_0', 'pred_asp_1', 'pred_asp_2', 'pred_asp_3', 'pred_asp_4', 'pred_aspect', 'pred_sentiment']:
        assert c in df_base.columns


if __name__ == '__main__':
    unittest.main()
