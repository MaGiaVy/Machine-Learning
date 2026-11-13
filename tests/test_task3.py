import os
import unittest
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

class TestTask3PhoBertMultiTask(unittest.TestCase):
    def setUp(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.nb_path = os.path.join(self.root, 'notebooks', '04_phobert_multitask.ipynb')
        self.models_dir = os.path.join(self.root, 'models')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')
        self.metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
        self.pred_path = os.path.join(self.proc_dir, 'test_pred_phobert.csv')

    def test_nb04_executed_and_no_fake_logs(self):
        """Kiểm tra NB04 mọi cell đều đã thực thi, không có training_logs hay hp_candidates gán cứng, có chia Eyeball/Blackbox Dev."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
        assert all(c.get('execution_count') is not None for c in code_cells), "NB04 has unexecuted cells"
        full_src = "\n".join("".join(c['source']) for c in code_cells)
        assert "training_logs = [" not in full_src, "Hardcoded training_logs still present in NB04"
        assert "hp_candidates" not in full_src, "Fake hp_candidates grid search still present in NB04"
        assert "df_eyeball" in full_src and "df_blackbox" in full_src, "Missing Eyeball/Blackbox Dev split in NB04"
        assert "ViTokenizer.tokenize" in full_src, "Missing ViTokenizer.tokenize in Colab fallback"

    def test_notebook_loss_weight_balanced(self):
        """Kiểm tra trọng số hàm loss được điều chỉnh sang 0.9 * loss_asp_fn."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        all_code = "\n".join("".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertIn("0.9 * loss_asp_fn", all_code, "Mã nguồn phải sử dụng hệ số 0.9 cho loss Aspect")
        self.assertNotIn("1.2 * loss_asp_fn", all_code, "Mã nguồn không được còn chứa hệ số 1.2 cho loss Aspect")

    def test_notebook_no_ground_truth_fallback(self):
        """Kiểm tra xóa bỏ triệt để việc fallback tự gán nhãn thật df_test['sentiment'].tolist()."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        all_code = "\n".join("".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertNotIn("df_test['sentiment'].tolist()", all_code, "Không được phép fallback nhãn thật sentiment")
        self.assertNotIn("df_test['aspect_group'].tolist()", all_code, "Không được phép fallback nhãn thật aspect")
        self.assertIn("raise FileNotFoundError", all_code, "Phải ném FileNotFoundError nếu không có checkpoint hoặc CSV")

    def test_test_pred_phobert_csv(self):
        """Kiểm tra test_pred_phobert.csv hợp lệ, đủ 2,224 dòng và không còn cờ is_no_content bắt nhầm câu tốt/tuyệt vời."""
        self.assertTrue(os.path.exists(self.pred_path), "File test_pred_phobert.csv không tồn tại")
        df = pd.read_csv(self.pred_path)
        self.assertEqual(len(df), 2224, "Số dòng tập test phải là 2,224")
        for col in ['pred_sentiment', 'pred_aspect', 'sentiment_conf', 'aspect_conf', 'needs_review', 'is_no_content']:
            self.assertIn(col, df.columns, f"Thiếu cột {col} trong test_pred_phobert.csv")
            self.assertFalse(df[col].isna().any(), f"Có giá trị NaN trong cột {col}")

        tot_rows = df[df['sentence'].str.contains('tốt|tuyệt vời', case=False, na=False)]
        self.assertGreater(len(tot_rows), 0, "Phải có câu chứa từ 'tốt' hoặc 'tuyệt vời' trong tập test")
        self.assertEqual((tot_rows['is_no_content'] == True).sum(), 0, "Không được có câu chứa 'tốt' hoặc 'tuyệt vời' bị đánh dấu is_no_content=True")

        short_praises = df[df['sentence_clean'].str.split().str.len() <= 5]
        short_positives = short_praises[(short_praises['n_star'] >= 4) & (short_praises['pred_sentiment'] == 2)]
        self.assertGreater(len(short_positives), 0, "Phải có câu khen ngắn tích cực")
        self.assertEqual(short_positives['needs_review'].sum(), 0, "Các câu khen 5 sao ngắn không được bị gắn cờ needs_review")

        sent_f1 = f1_score(df['sentiment'], df['pred_sentiment'], average='macro')
        asp_f1 = f1_score(df['aspect_group'], df['pred_aspect'], average='macro')
        self.assertGreater(sent_f1, 0.70, f"PhoBERT Sentiment Macro F1 ({sent_f1:.4f}) phải > 0.70")
        self.assertGreater(asp_f1, 0.70, f"PhoBERT Aspect Macro F1 ({asp_f1:.4f}) phải > 0.70")

    def test_model_metadata_json(self):
        """Kiểm tra model_metadata.json đúng cú pháp, khớp 100% với test_pred_baseline.csv và test_pred_phobert.csv."""
        self.assertTrue(os.path.exists(self.metadata_path), "File model_metadata.json không tồn tại")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn("baseline_tfidf_linearsvc", data)
        self.assertIn("phobert_multitask", data)

        df_base = pd.read_csv(os.path.join(self.proc_dir, 'test_pred_baseline.csv'))
        df_pho = pd.read_csv(self.pred_path)

        base = data["baseline_tfidf_linearsvc"]
        self.assertEqual(base["sentiment_macro_f1"], round(f1_score(df_base['sentiment'], df_base['pred_sentiment'], average='macro'), 4))
        self.assertEqual(base["sentiment_accuracy"], round(accuracy_score(df_base['sentiment'], df_base['pred_sentiment']), 4))
        self.assertEqual(base["aspect_macro_f1"], round(f1_score(df_base['aspect_group'], df_base['pred_aspect'], average='macro'), 4))
        self.assertEqual(base["aspect_accuracy"], round(accuracy_score(df_base['aspect_group'], df_base['pred_aspect']), 4))

        phobert = data["phobert_multitask"]
        self.assertEqual(phobert["sentiment_macro_f1"], round(f1_score(df_pho['sentiment'], df_pho['pred_sentiment'], average='macro'), 4))
        self.assertEqual(phobert["sentiment_accuracy"], round(accuracy_score(df_pho['sentiment'], df_pho['pred_sentiment']), 4))
        self.assertEqual(phobert["aspect_macro_f1"], round(f1_score(df_pho['aspect_group'], df_pho['pred_aspect'], average='macro'), 4))
        self.assertEqual(phobert["aspect_accuracy"], round(accuracy_score(df_pho['aspect_group'], df_pho['pred_aspect']), 4))


def test_nb04_executed_and_no_fake_logs():
    with open('notebooks/04_phobert_multitask.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    assert all(c.get('execution_count') is not None for c in code_cells), "NB04 has unexecuted cells"
    full_src = "\n".join("".join(c['source']) for c in code_cells)
    assert "training_logs = [" not in full_src, "Hardcoded training_logs still present in NB04"

if __name__ == '__main__':
    unittest.main()
