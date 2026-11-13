import os
import json
import unittest
import pandas as pd
from PIL import Image

class TestTask4ActionableInsights(unittest.TestCase):
    def setUp(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.nb_path = os.path.join(self.root, 'notebooks', '05_shap_and_actionable_insights.ipynb')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')
        self.fig_dir = os.path.join(self.root, 'reports', 'figures')
        self.csv_path = os.path.join(self.proc_dir, 'actionable_insights.csv')
        self.fig_path = os.path.join(self.fig_dir, 'shap_top_negative_words.png')

    def test_actionable_insights_full_coverage_and_positive_shap(self):
        """Kiểm tra độ bao phủ 100% của Actionable Insights và SHAP Local chỉ lấy giá trị dương (sv > 0)."""
        df_pred = pd.read_csv(os.path.join(self.proc_dir, 'test_pred_phobert.csv'))
        df_ins = pd.read_csv(self.csv_path)
        total_neg = int((df_pred['pred_sentiment'] == 0).sum())
        assert df_ins['negative_count'].sum() == total_neg, (
            f"Insights coverage mismatch: {df_ins['negative_count'].sum()} != {total_neg}"
        )
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb_src = f.read()
        assert '-abs(x[1])' not in nb_src, "Local SHAP explanation still uses -abs(x[1])"
        assert 'sv_neg_one[i] > 0' in nb_src, "Local SHAP must filter strictly positive contributions sv_neg_one[i] > 0"

    def test_notebook_filtering_logic(self):
        """Kiểm tra mã nguồn Notebook 05 không còn chứa 'else df_neg' và lọc chính xác theo asp_id."""
        self.assertTrue(os.path.exists(self.nb_path), f"Notebook không tồn tại: {self.nb_path}")
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        all_code = "\n".join("".join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertNotIn("else df_neg", all_code, "Notebook 05 không được còn chứa logic 'else df_neg'")
        self.assertIn("sub = df_neg[df_neg['pred_aspect'] == asp_id]", all_code,
                      "Notebook 05 phải lọc tập con đúng theo asp_id cho tất cả các nhóm hành động")

    def test_actionable_insights_csv(self):
        """Kiểm tra file actionable_insights.csv tồn tại, không rỗng, chứa đủ 6 nhóm hành động MECE hợp lệ."""
        self.assertTrue(os.path.exists(self.csv_path), f"File không tồn tại: {self.csv_path}")
        self.assertGreater(os.path.getsize(self.csv_path), 0, "File actionable_insights.csv bị rỗng")

        df = pd.read_csv(self.csv_path)
        expected_cols = [
            'cluster_id', 'aspect_cluster', 'negative_count',
            'matched_keywords', 'sample_reviews', 'recommended_action'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Thiếu cột {col} trong actionable_insights.csv")
            self.assertFalse(df[col].isna().any(), f"Cột {col} chứa giá trị NaN")

        self.assertEqual(len(df), 6, f"Dự kiến 6 nhóm hành động, thực tế có {len(df)}")

        expected_clusters = {
            'Hiệu năng & Cấu hình',
            'Thời lượng Pin & Sạc',
            'Màn hình & Camera - Nhóm Camera',
            'Màn hình & Camera - Nhóm Màn hình',
            'Giá cả & Hậu mãi',
            'Thiết kế & Tổng quan'
        }
        self.assertEqual(set(df['aspect_cluster']), expected_clusters, "Các nhóm cụm hành động không khớp danh mục MECE chuẩn")

        for _, row in df.iterrows():
            self.assertGreater(row['negative_count'], 0, f"Cụm {row['aspect_cluster']} có negative_count <= 0")
            self.assertTrue(isinstance(row['sample_reviews'], str) and len(row['sample_reviews'].strip()) > 0,
                            f"Cụm {row['aspect_cluster']} thiếu sample_reviews")
            self.assertTrue(isinstance(row['recommended_action'], str) and len(row['recommended_action'].strip()) > 0,
                            f"Cụm {row['aspect_cluster']} thiếu recommended_action")

    def test_shap_top_negative_words_figure(self):
        """Kiểm tra file hình ảnh biểu đồ SHAP reports/figures/shap_top_negative_words.png tồn tại và hợp lệ."""
        self.assertTrue(os.path.exists(self.fig_path), f"File biểu đồ không tồn tại: {self.fig_path}")
        self.assertGreater(os.path.getsize(self.fig_path), 5000, "Kích thước file ảnh biểu đồ quá nhỏ hoặc bị hỏng")
        with Image.open(self.fig_path) as img:
            img.verify()
            self.assertEqual(img.format, 'PNG', "Định dạng ảnh phải là PNG")


def test_actionable_insights_full_coverage_and_positive_shap():
    df_pred = pd.read_csv('data/processed/test_pred_phobert.csv')
    df_ins = pd.read_csv('data/processed/actionable_insights.csv')
    total_neg = int((df_pred['pred_sentiment'] == 0).sum())
    assert df_ins['negative_count'].sum() == total_neg, (
        f"Insights coverage mismatch: {df_ins['negative_count'].sum()} != {total_neg}"
    )
    with open('notebooks/05_shap_and_actionable_insights.ipynb', encoding='utf-8') as f:
        nb_src = f.read()
    assert '-abs(x[1])' not in nb_src, "Local SHAP explanation still uses -abs(x[1])"

if __name__ == '__main__':
    unittest.main()
