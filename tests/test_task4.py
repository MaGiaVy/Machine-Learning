import json
import os
import sys
import unittest

import pandas as pd
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.labels import ABSA_STATES


class TestTask4ShapAndActionableInsightsV4(unittest.TestCase):
    def setUp(self):
        self.root = ROOT_DIR
        self.nb_path = os.path.join(self.root, 'notebooks', '05_shap_and_actionable_insights.ipynb')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')
        self.fig_dir = os.path.join(self.root, 'reports', 'figures')
        self.csv_path = os.path.join(self.proc_dir, 'actionable_insights.csv')
        self.fig_path = os.path.join(self.fig_dir, 'shap_top_negative_words.png')

    def test_nb05_executed_neg_idx_and_no_camera_regex(self):
        """Kiểm tra NB05 đã chạy hết cell, dùng ABSA_STATES.index('Negative'), count>=10, Occlusion XAI và bỏ regex Camera/Màn hình."""
        self.assertTrue(os.path.exists(self.nb_path))
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
        self.assertTrue(all(c.get('execution_count') is not None for c in code_cells), "NB05 còn cell chưa thực thi")
        all_code = "\n".join("".join(c.get('source', [])) for c in code_cells)
        self.assertIn("ABSA_STATES.index('Negative')", all_code)
        self.assertIn("counts >= 10", all_code)
        self.assertIn("sv_neg_one[i] > 0", all_code)
        self.assertIn("explain_phobert_occlusion", all_code)
        self.assertNotIn("cam_pat", all_code, "NB05 phải bỏ regex tách Camera/Màn hình thủ công")
        self.assertNotIn("[:400]", all_code, "NB05 phải tính SHAP trên toàn bộ 2,224 mẫu Test")

    def test_actionable_insights_5_clusters_and_unique_coverage(self):
        """Kiểm tra actionable_insights.csv có đúng 5 cụm khía cạnh, tổng lượt chê khớp pred_absa_j==1 và có extracted_keywords."""
        self.assertTrue(os.path.exists(self.csv_path))
        df_ins = pd.read_csv(self.csv_path)
        df_pred = pd.read_csv(os.path.join(self.proc_dir, 'test_pred_phobert.csv'), keep_default_na=False)
        neg_idx = ABSA_STATES.index('Negative')
        pred_cols = [f'pred_absa_{j}' for j in range(5)]
        pred_mat = df_pred[pred_cols].values.astype(int)

        total_neg_mentions = int((pred_mat == neg_idx).sum())
        df_neg = df_pred[(pred_mat == neg_idx).any(axis=1)]
        union_idx = set().union(*[set(df_pred.index[df_pred[f'pred_absa_{j}'].astype(int) == neg_idx]) for j in range(5)])
        self.assertEqual(len(union_idx), len(df_neg))
        self.assertEqual(int(df_ins['negative_count'].sum()), total_neg_mentions)

        expected_cols = [
            'cluster_id', 'aspect_cluster', 'negative_count',
            'extracted_keywords', 'matched_keywords', 'sample_reviews', 'recommended_action',
        ]
        for col in expected_cols:
            self.assertIn(col, df_ins.columns)
            self.assertFalse(df_ins[col].isna().any())

        self.assertEqual(len(df_ins), 5)
        expected_clusters = {
            'Hiệu năng & Cấu hình',
            'Màn hình & Camera',
            'Thời lượng Pin & Sạc',
            'Giá cả & Hậu mãi',
            'Thiết kế & Tổng quan',
        }
        self.assertEqual(set(df_ins['aspect_cluster']), expected_clusters)
        for _, row in df_ins.iterrows():
            self.assertGreater(int(row['negative_count']), 0)
            self.assertGreater(len(str(row['extracted_keywords']).strip()), 0)

    def test_shap_top_negative_words_figure(self):
        """Kiểm tra biểu đồ reports/figures/shap_top_negative_words.png tồn tại và hợp lệ."""
        self.assertTrue(os.path.exists(self.fig_path))
        self.assertGreater(os.path.getsize(self.fig_path), 5000)
        with Image.open(self.fig_path) as img:
            img.verify()
            self.assertEqual(img.format, 'PNG')


if __name__ == '__main__':
    unittest.main()
