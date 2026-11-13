# PhoBERT Multi-Task Model Weights (UIT-ViSFD)

File trọng số huấn luyện `phobert_multitask.pt` (~540 MB) vượt quá giới hạn lưu trữ Git.
Khi chạy trong môi trường Production hoặc Google Colab:
- Tokenizer files đã được tích hợp sẵn trong thư mục này (`vocab.txt`, `bpe.codes`, `tokenizer_config.json`).
- Trọng số mô hình tải từ Google Drive / S3 hoặc huấn luyện trực tiếp bằng GPU Google Colab T4.
- Kết quả kiểm chuẩn đánh giá niêm phong Test set lưu tại `data/processed/test_pred_phobert.csv`.
