import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.labels import ABSA_COLS, compute_absa_metrics, compute_paired_bootstrap_ci


class TestTask3PhoBertABSAv4(unittest.TestCase):
    def setUp(self):
        self.root = ROOT_DIR
        self.nb_path = os.path.join(self.root, 'notebooks', '04_phobert_multitask.ipynb')
        self.models_dir = os.path.join(self.root, 'models')
        self.save_dir = os.path.join(self.models_dir, 'phobert_absa_v4')
        self.proc_dir = os.path.join(self.root, 'data', 'processed')
        self.metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
        self.pred_path = os.path.join(self.proc_dir, 'test_pred_phobert.csv')
        self.dev_pred_path = os.path.join(self.proc_dir, 'dev_pred_phobert.csv')

    def test_nb04_executed_and_v4_design(self):
        """Kiểm tra NB04 đã thực thi toàn bộ cell, có seed_everything(42), chia Eyeball/Calib/Val và dùng MAX_LEN_ABSA=128."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
        self.assertTrue(all(c.get('execution_count') is not None for c in code_cells), "NB04 còn cell chưa thực thi")
        full_src = "\n".join("".join(c['source']) for c in code_cells)
        self.assertNotIn("training_logs = [", full_src)
        self.assertNotIn("hp_candidates", full_src)
        self.assertIn("df_eyeball", full_src)
        self.assertIn("df_calib", full_src)
        self.assertIn("df_val", full_src)
        self.assertIn("seed_everything(42)", full_src)
        self.assertIn("PhoBertABSA", full_src)
        self.assertIn("MAX_LEN_ABSA", full_src)
        self.assertIn("ABSATemperatureScaler", full_src)
        self.assertIn("compute_paired_bootstrap_ci", full_src)
        self.assertIn("raise FileNotFoundError", full_src)

    def test_test_and_dev_pred_phobert_csv(self):
        """Kiểm tra test_pred_phobert.csv (2224) và dev_pred_phobert.csv (1110) đủ cột ABSA 5x4 và mức độ khiếu nại ngầm."""
        for path, exp_len in [(self.pred_path, 2224), (self.dev_pred_path, 1110)]:
            self.assertTrue(os.path.exists(path), f"Thiếu file {path}")
            df = pd.read_csv(path, keep_default_na=False)
            self.assertEqual(len(df), exp_len)
            pred_cols = [f'pred_absa_{j}' for j in range(5)]
            conf_cols = [f'absa_conf_{j}' for j in range(5)]
            req = pred_cols + conf_cols + [
                'pred_absa_derived_sent', 'min_aspect_conf', 'So_Khia_Canh_Tieu_Cuc',
                'Muc_Do_Khieu_Nai_Ngam', 'Khieu_Nai_Ngam_4_5_Sao', 'needs_review', 'is_no_content',
            ] + ABSA_COLS
            for c in req:
                self.assertIn(c, df.columns, f"Thiếu cột {c} trong {path}")
            for c in pred_cols:
                vals = set(df[c].astype(int).unique())
                self.assertTrue(vals.issubset({0, 1, 2, 3}), f"Cột {c} ngoài miền 0..3: {vals}")
            self.assertTrue(set(df['Muc_Do_Khieu_Nai_Ngam'].unique()).issubset({'N/A', 'Không', 'Nhẹ', 'Cao'}))
            is_low_star = df['n_star'].astype(int) <= 3
            self.assertTrue((df.loc[is_low_star, 'Muc_Do_Khieu_Nai_Ngam'] == 'N/A').all())
            self.assertTrue(
                (df['Khieu_Nai_Ngam_4_5_Sao'].astype(str).str.lower() == (df['Muc_Do_Khieu_Nai_Ngam'] == 'Cao').astype(str).str.lower()).all()
            )

        df_te = pd.read_csv(self.pred_path, keep_default_na=False)
        tot_rows = df_te[df_te['sentence'].str.contains('tốt|tuyệt vời', case=False, na=False)]
        self.assertGreater(len(tot_rows), 0)
        self.assertEqual((tot_rows['is_no_content'].astype(str).str.lower() == 'true').sum(), 0)

    def test_metadata_matches_recomputed_metrics_and_bootstrap_ci(self):
        """Kiểm tra model_metadata.json khớp chỉ số tính lại từ test_pred_phobert.csv, có lát cắt xung đột và paired bootstrap CI."""
        self.assertTrue(os.path.exists(self.metadata_path))
        meta = json.load(open(self.metadata_path, encoding='utf-8'))
        self.assertEqual(meta.get('max_len'), 128)
        self.assertIn('phobert_absa_v4', meta)
        self.assertIn('baseline_absa_v4', meta)
        self.assertIn('calibration', meta)
        self.assertEqual(len(meta['calibration'].get('temps', [])), 5)

        df_te = pd.read_csv(self.pred_path, keep_default_na=False)
        y_true = df_te[ABSA_COLS].values.astype(int)
        y_pred = df_te[[f'pred_absa_{j}' for j in range(5)]].values.astype(int)
        recomp = compute_absa_metrics(y_true, y_pred)

        p_meta = meta['phobert_absa_v4']
        for k in ['absa_micro_f1', 'absa_macro_f1', 'mean_4cls_macro_f1', 'polarity_given_detected_macro_f1', 'majority_4cls_macro_f1']:
            self.assertIn(k, p_meta)
            self.assertAlmostEqual(float(p_meta[k]), float(recomp[k]), places=3, msg=f"Lệch {k}")

        self.assertIn('conflict_absa_macro_f1', p_meta)
        self.assertIn('conflict_with_neg_absa_macro_f1', p_meta)
        self.assertIn('silent_complaint_dist_test', p_meta)

        ci = p_meta.get('paired_bootstrap_ci_vs_baseline')
        self.assertIsNotNone(ci, "Thiếu paired_bootstrap_ci_vs_baseline trong model_metadata.json")
        for k in ['a_mean', 'a_ci95', 'b_mean', 'b_ci95', 'diff_mean', 'diff_ci95']:
            self.assertIn(k, ci)
        print(
            f"\n[SO SÁNH ABSA 5x4 TEST] PhoBERT absa_macro_f1={p_meta['absa_macro_f1']:.4f} "
            f"vs Baseline={meta['baseline_absa_v4']['absa_macro_f1']:.4f} | "
            f"Diff={ci['diff_mean']:+.4f} (95% CI: {ci['diff_ci95']})"
        )


if __name__ == '__main__':
    unittest.main()
