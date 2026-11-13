import os
import re
import shutil
import sys
import tempfile
import unittest

import openpyxl
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.excel_ingestion import (
    build_clean_table,
    detect_header_row,
    detect_star_column,
    detect_text_columns,
    mask_tabular_pii,
    pick_best_sheet,
    read_any_file,
    read_csv_safe,
    read_xls_safe,
)

MESSY_DIR = os.path.join(ROOT_DIR, 'data', 'samples_messy')
TMP_BASE = os.path.join(ROOT_DIR, '.tmp_run')


class TestTask6ExcelIngestion(unittest.TestCase):
    def setUp(self):
        os.makedirs(TMP_BASE, exist_ok=True)
        self._tmp_dir = tempfile.mkdtemp(dir=TMP_BASE)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_05_pos_legacy_is_real_biff_xls_and_10_is_html(self):
        p05 = os.path.join(MESSY_DIR, '05_pos_legacy.xls')
        with open(p05, 'rb') as f:
            magic = f.read(8)
        self.assertEqual(magic, b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')

        xls_sheets = read_xls_safe(p05)
        self.assertIn('Sheet1', xls_sheets)
        df05 = xls_sheets['Sheet1']
        self.assertEqual(len(df05), 25)
        self.assertEqual(list(df05.columns), ['Ma_DH', 'Khach_Hang', 'Nhan_Xet', 'Diem_Sao', 'Ngay'])

        df05_xlrd = pd.read_excel(p05, engine='xlrd')
        self.assertEqual(len(df05_xlrd), 25)

        p10 = os.path.join(MESSY_DIR, '10_html_table_as_xls.xls')
        sheets10, meta10 = read_any_file(p10)
        self.assertEqual(meta10['format'], 'html_xls')
        self.assertEqual(len(sheets10['Sheet1']), 25)

    def test_09_merged_cells_pagination_authentic_and_clean(self):
        p09 = os.path.join(MESSY_DIR, '09_merged_cells_pagination.xlsx')
        wb = openpyxl.load_workbook(p09, data_only=True)
        try:
            ws = wb['DanhSach']
            merged_strs = [str(r) for r in ws.merged_cells.ranges]
            self.assertGreaterEqual(len(merged_strs), 5)
            # Non-top cell in first merged category block must genuinely be None in openpyxl
            self.assertIsNone(ws.cell(5, 2).value)
            # Star column must not be merged
            for r in ws.merged_cells.ranges:
                if r.min_row >= 4:
                    self.assertNotIn(5, range(r.min_col, r.max_col + 1))
            # Repeated mid-page header row exists after row 14
            row19 = [ws.cell(19, c).value for c in range(1, 7)]
            self.assertEqual(row19, ['STT', 'Nhóm sản phẩm', 'Khách hàng', 'Phản hồi sản phẩm', 'Số sao đánh giá', 'Ghi chú'])
        finally:
            wb.close()

        sheets, meta = read_any_file(p09)
        self.assertIn('DanhSach', meta['merged_ranges'])
        self.assertGreaterEqual(len(meta['merged_ranges']['DanhSach']), 5)

        sname, raw_df = pick_best_sheet(sheets)
        self.assertEqual(sname, 'DanhSach')
        hdr = detect_header_row(raw_df)
        self.assertEqual(hdr, 2)

        clean_df = build_clean_table(raw_df, header_row=hdr, meta=meta, sheet_name=sname)
        self.assertEqual(len(clean_df), 25)
        self.assertFalse((clean_df['STT'].astype(str) == 'STT').any())
        self.assertEqual(clean_df['Nhóm sản phẩm'].isna().sum(), 0)
        self.assertEqual(clean_df['Nhóm sản phẩm'].value_counts().to_dict(), {
            'Điện thoại Flagship': 5,
            'Điện thoại Tầm trung': 5,
            'Phụ kiện & Sạc nhanh': 5,
            'Điện thoại Phổ thông': 5,
            'Dịch vụ & Hậu mãi': 5,
        })

    def test_exact_unmerge_no_blind_ffill_on_blank_phone_or_star(self):
        xlsx_path = os.path.join(self._tmp_dir, 'synthetic_merged.xlsx')
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'Reviews'
        headers = ['STT', 'Nhóm sản phẩm', 'SĐT', 'Phản hồi', 'Số sao']
        ws.append(headers)
        ws.append([1, 'Điện thoại', '0901234567', 'Máy mượt pin trâu dùng rất ổn', 5])
        ws.append([2, None, None, 'Camera chụp đẹp lắm nhưng sạc hơi ấm', None])
        ws.append([3, None, '0987654321', 'Giao hàng nhanh đóng gói rất cẩn thận', 4])
        ws.append(headers)  # repeated mid-page header at row 5
        ws.append([4, 'Phụ kiện', None, 'Cáp sạc chắc chắn truyền dữ liệu nhanh', None])
        ws.append([5, None, '0911223344', 'Ốp lưng đẹp cầm vừa tay rất thích', 5])
        ws.merge_cells('B2:B4')
        ws.merge_cells('B6:B7')
        wb.save(xlsx_path)
        wb.close()

        sheets, meta = read_any_file(xlsx_path)
        sname, raw_df = pick_best_sheet(sheets)
        clean_df = build_clean_table(raw_df, meta=meta, sheet_name=sname)

        self.assertEqual(len(clean_df), 5)
        self.assertEqual(
            clean_df['Nhóm sản phẩm'].tolist(),
            ['Điện thoại', 'Điện thoại', 'Điện thoại', 'Phụ kiện', 'Phụ kiện'],
        )
        # Blank phone and blank star rows must NOT be blindly forward-filled
        self.assertTrue(pd.isna(clean_df.loc[1, 'SĐT']))
        self.assertTrue(pd.isna(clean_df.loc[1, 'Số sao']))
        self.assertTrue(pd.isna(clean_df.loc[3, 'SĐT']))
        self.assertTrue(pd.isna(clean_df.loc[3, 'Số sao']))

    def test_utf8_tsv_customer_sentiment_note_and_star_edge_cases(self):
        tsv_path = os.path.join(self._tmp_dir, 'utf8_reviews.tsv')
        rows = [
            'Order_SN\tBuyer\tCustomer_Sentiment_Note\tTại sao không mua',
            'ORD101\tNguyễn An\tSản phẩm tốt, màn đẹp, pin ổn, giao nhanh, rất hài lòng\tKhông có vấn đề gì',
            'ORD102\tTrần Bình\tMáy chạy mượt, loa to, camera rõ, giá hợp lý, đáng mua\tGiá hơi cao xíu',
            'ORD103\tLê Chi\tPin tụt nhanh, máy nóng, sạc chậm, cần cải thiện thêm\tChờ bản mới hơn',
            'ORD104\tPhạm Dũng\tThiết kế đẹp, cầm đầm tay, màu sang, đóng gói kỹ\tKhông có lý do',
        ]
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows))

        sheets = read_csv_safe(tsv_path)
        df_tsv = sheets['Sheet1']
        self.assertEqual(len(df_tsv), 4)
        self.assertEqual(
            list(df_tsv.columns),
            ['Order_SN', 'Buyer', 'Customer_Sentiment_Note', 'Tại sao không mua'],
        )

        clean_tsv = build_clean_table(df_tsv)
        txt_cols = detect_text_columns(clean_tsv)
        self.assertIn('Customer_Sentiment_Note', txt_cols)
        self.assertIsNone(detect_star_column(clean_tsv))

        # Verify Review_Text, combined_review, and Ý kiến đóng gói giao hàng are never blocked
        df_tok = pd.DataFrame({
            'STT': [1, 2, 3, 4],
            'Nhóm sản phẩm': ['Nhóm điện thoại cao cấp mới ra mắt', 'Nhóm điện thoại cao cấp mới ra mắt', 'Nhóm điện thoại cao cấp mới ra mắt', 'Nhóm điện thoại cao cấp mới ra mắt'],
            'Khu vực': ['Khu vực miền Nam và Tây Nam Bộ', 'Khu vực miền Bắc và Đông Bắc Bộ', 'Khu vực miền Trung và Tây Nguyên', 'Khu vực TP Hồ Chí Minh'],
            'Review_Text': ['Máy dùng mượt pin trâu', 'Camera đẹp lắm nha', 'Sạc nhanh ổn áp', 'Màn hình sắc nét'],
            'combined_review': ['Máy dùng mượt pin trâu', 'Camera đẹp lắm nha', 'Sạc nhanh ổn áp', 'Màn hình sắc nét'],
            'Ý kiến đóng gói giao hàng': ['Đóng gói kỹ 3 lớp xốp', 'Giao hàng nhanh đúng hẹn', 'Hộp nguyên seal không móp', 'Shipper thân thiện nhiệt tình'],
            'Điểm hài lòng': [5, 4, 5, 3],
        })
        det_txt = detect_text_columns(df_tok)
        self.assertIn('Review_Text', det_txt)
        self.assertIn('combined_review', det_txt)
        self.assertIn('Ý kiến đóng gói giao hàng', det_txt)
        self.assertNotIn('Nhóm sản phẩm', det_txt)
        self.assertNotIn('Khu vực', det_txt)
        self.assertNotIn('STT', det_txt)
        self.assertEqual(detect_star_column(df_tok), 'Điểm hài lòng')

        # Numeric 1..5 fallback without star keyword
        df_num = pd.DataFrame({
            'Comment': ['Sản phẩm dùng rất tốt ạ', 'Pin hơi yếu một chút', 'Màn hình đẹp lắm', 'Giao nhanh ok'],
            'Danh_Gia_Num': [5, 3, 5, 4],
        })
        self.assertEqual(detect_star_column(df_num), 'Danh_Gia_Num')

    def test_all_10_messy_files_pass(self):
        truth = {
            '01_shopee_clean.xlsx': {'text': ['Đánh giá của khách'], 'star': 'Số sao (1-5)'},
            '02_tiki_semicolon_bom.csv': {'text': ['Review_Content'], 'star': 'Rating_Star'},
            '03_lazada_english.xlsx': {'text': ['Comment / Review'], 'star': 'Rating'},
            '04_tiktok_title_offset.xlsx': {'text': ['Ý kiến đánh giá'], 'star': 'Số sao'},
            '05_pos_legacy.xls': {'text': ['Nhan_Xet'], 'star': 'Diem_Sao'},
            '06_multi_sheet_report.xlsx': {'text': ['Nội dung phản hồi'], 'star': 'Đánh giá sao'},
            '07_utf16_tab_export.csv': {'text': ['Review_Text'], 'star': 'Stars'},
            '08_two_text_cols.xlsx': {'text': ['Nhận xét về máy', 'Ý kiến đóng gói giao hàng'], 'star': 'Sao'},
            '09_merged_cells_pagination.xlsx': {'text': ['Phản hồi sản phẩm'], 'star': 'Số sao đánh giá'},
            '10_html_table_as_xls.xls': {'text': ['Nội Dung Đánh Giá'], 'star': 'Đánh Giá Sao'},
        }
        passed = 0
        for fname, exp in sorted(truth.items()):
            sheets, meta = read_any_file(os.path.join(MESSY_DIR, fname))
            sname, raw_df = pick_best_sheet(sheets)
            hdr = detect_header_row(raw_df)
            tab = build_clean_table(raw_df, header_row=hdr, meta=meta, sheet_name=sname)
            det_txt = detect_text_columns(tab)
            det_star = detect_star_column(tab)
            self.assertEqual(len(tab), 25, f"{fname} row count mismatch: {len(tab)}")
            for t in exp['text']:
                self.assertIn(t, det_txt, f"{fname} missing text col {t}; got {det_txt}")
            self.assertEqual(det_star, exp['star'], f"{fname} star mismatch: {det_star}")
            self.assertTrue((tab['review_text'].str.len() > 0).all(), f"{fname} has empty review_text")
            if fname == '08_two_text_cols.xlsx':
                first_expected = f"{tab.loc[0, 'Nhận xét về máy']}. {tab.loc[0, 'Ý kiến đóng gói giao hàng']}"
                self.assertEqual(tab.loc[0, 'review_text'], first_expected)
            passed += 1
        self.assertEqual(passed, 10)

    def test_mask_tabular_pii(self):
        df = pd.DataFrame({
            'Mã đơn': ['ORD111222', ''],
            'Order_Code': ['DH333444', None],
            'Order ID': ['ORD555666', 'ORD777888'],
            'Ma_DH': ['MDH001', 'MDH002'],
            'Order_SN': ['SN999000', 'SN888000'],
            'Tên người mua': ['Nguyễn Văn A', None],
            'Khách hàng': ['Trần Thị B', 'Lê Văn C'],
            'Buyer': ['buyer_1', 'buyer_2'],
            'Username': ['user_x', ''],
            'Họ tên': ['Phạm D', 'Hoàng E'],
            'SĐT': ['0901234567', None],
            'Số điện thoại': ['0987654321', '0912345678'],
            'Phone': ['0779887766', '0868112233'],
            'Đánh giá của khách': [
                'Đơn ORD111222 gọi 0901234567 không ai nghe',
                'Sản phẩm rất tốt giao hàng nhanh chóng',
            ],
            'Số sao': [1, 5],
        })
        df['review_text'] = df['Đánh giá của khách']

        def clean_fn(txt: str) -> str:
            txt = re.sub(r'\b(?:ORD|DH)[0-9A-Z]+\b', '[ORDER_ID]', txt)
            txt = re.sub(r'\b0[0-9]{9}\b', '[PHONE]', txt)
            return txt

        masked_meta_only = mask_tabular_pii(df)
        self.assertEqual(masked_meta_only.loc[0, 'Tên người mua'], '[PERSON]')
        self.assertTrue(pd.isna(masked_meta_only.loc[1, 'Tên người mua']))
        self.assertEqual(masked_meta_only.loc[0, 'Khách hàng'], '[PERSON]')
        self.assertEqual(masked_meta_only.loc[0, 'Buyer'], '[PERSON]')
        self.assertEqual(masked_meta_only.loc[0, 'Username'], '[PERSON]')
        self.assertEqual(masked_meta_only.loc[1, 'Username'], '')
        self.assertEqual(masked_meta_only.loc[0, 'Họ tên'], '[PERSON]')
        self.assertEqual(masked_meta_only.loc[0, 'SĐT'], '[PHONE]')
        self.assertTrue(pd.isna(masked_meta_only.loc[1, 'SĐT']))
        self.assertEqual(masked_meta_only.loc[0, 'Số điện thoại'], '[PHONE]')
        self.assertEqual(masked_meta_only.loc[0, 'Phone'], '[PHONE]')
        self.assertEqual(masked_meta_only.loc[0, 'Mã đơn'], '[ORDER_ID]')
        self.assertEqual(masked_meta_only.loc[0, 'Order_Code'], '[ORDER_ID]')
        self.assertEqual(masked_meta_only.loc[0, 'Order ID'], '[ORDER_ID]')
        self.assertEqual(masked_meta_only.loc[0, 'Ma_DH'], '[ORDER_ID]')
        self.assertEqual(masked_meta_only.loc[0, 'Order_SN'], '[ORDER_ID]')
        self.assertEqual(masked_meta_only['Số sao'].tolist(), [1, 5])

        masked_full = mask_tabular_pii(df, text_clean_fn=clean_fn)
        for col in ['Đánh giá của khách', 'review_text']:
            self.assertNotIn('0901234567', masked_full.loc[0, col])
            self.assertNotIn('ORD111222', masked_full.loc[0, col])
            self.assertIn('[PHONE]', masked_full.loc[0, col])
            self.assertIn('[ORDER_ID]', masked_full.loc[0, col])


if __name__ == '__main__':
    unittest.main()
