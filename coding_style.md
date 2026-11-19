# Hướng dẫn phong cách viết code Data Science (Dự án nhóm 4 người)

> Nguyên tắc chung: **cell gọn, kiểm tra dữ liệu liên tục sau mỗi bước, tái sử dụng qua `src/`, tuyệt đối không để rò rỉ dữ liệu**. Viết tự nhiên, dễ đọc cho cả 4 thành viên khi ghép code, không rườm rà.

---

## 1. Cấu trúc dự án & notebook

- **Tách logic dùng chung vào `src/`:** hàm xử lý nhãn (`src/labels.py`), làm sạch văn bản (`src/text_cleaner.py`), mô hình & dataset (`src/modeling.py`), ghi log (`src/exp_log.py`). Không copy-paste cùng một hàm vào nhiều notebook.
- **Pipeline chính (`notebooks/00–06`):** chia theo giai đoạn, trao đổi giữa các bước bằng file CSV trong `data/processed/`.
- **Thực nghiệm (`experiments/`):** mỗi file kiểm tra 1 câu hỏi độc lập, ghi kết quả về `reports/experiments_log.csv`.
- **Tương thích môi trường:** tuân theo [RULE.md](RULE.md) để chạy được trên Local, Google Colab và Kaggle GPU (dùng `os.path.join`, tự phát hiện `ROOT`/`WORK_DIR`, `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

---

## 2. Nhịp viết notebook (tự nhiên, dễ kiểm soát)

- **Cell vừa phải, mỗi cell một việc:** thường 3–15 dòng cho một bước biến đổi hoặc một bảng/biểu đồ kiểm tra.
- **Thử trên 1 mẫu trước khi chạy cả bảng:** lấy 1 dòng mẫu (`sample_text`) chạy thử hàm làm sạch/tách từ, nhìn thấy output đúng rồi mới `.apply(...)` toàn cột.
- **Kiểm tra dữ liệu liên tục:** ngay sau mỗi phép biến đổi hoặc lọc trùng, gọi nhanh `.shape`, `.info()`, `.head()`, `.value_counts()` ở cell riêng để bắt sớm lỗi mất dòng hoặc sai kiểu.
- **Import gọn gàng:** thư viện cốt lõi (`os`, `sys`, `pandas`, `numpy`) đặt ở đầu; thư viện riêng của từng mô hình có thể import ngay trước phần đó cho dễ đọc.

---

## 3. Đặt tên & Comment

- **Tên biến rõ ngữ cảnh:** ưu tiên tên ngắn nhưng phân biệt được tập và trạng thái (ví dụ `df_train`, `df_dev`, `X_tr`, `y_dev_sent`, `lr_sent_pipe`), tránh dùng chung 1 tên `tempt` hay `df` cho nhiều thứ dễ ghi đè nhầm khi chạy lại cell.
- **Comment bằng tiếng Việt ngắn, nêu *lý do* thay vì nhắc lại cú pháp:**
  - Nên: `#Dung Pipeline de fit TF-IDF rieng trong tung fold CV, tranh data leakage`
  - Không nên: `#Khoi tao TfidfVectorizer` (hiển nhiên, không cần viết).
- **Markdown ngắn đánh dấu mốc:** dùng tiêu đề rõ ràng (`## 1. LÀM SẠCH DỮ LIỆU`, `## 2. ĐÁNH GIÁ TRÊN DEV`), có 1–2 câu tóm tắt kết luận dưới bảng/biểu đồ thay vì đoạn văn dài dòng.

---

## 4. Quy tắc vàng cho EDA & Mô hình (chuẩn Data Scientist)

1. **Chống Data Leakage tuyệt đối:**
   - Mọi bước EDA, `TfidfVectorizer`, scaler, chọn siêu tham số (`GridSearchCV`), early stopping và chỉnh ngưỡng đều **chỉ nhìn `train` và `dev`** (trong CV phải bọc trong `sklearn.pipeline.Pipeline`).
   - Tập **`test` chỉ mở 1 lần cuối cùng** ở pipeline chính sau khi đã chốt mô hình, không dùng `test` để chọn tham số qua nhiều vòng thử.
2. **Đi từ đơn giản đến phức tạp:** luôn chạy baseline nhẹ (`TF-IDF + LogisticRegression`) làm mốc trước khi dùng Deep Learning (`PhoBERT`), so sánh cả **Train vs CV/Dev** để phát hiện overfitting.
3. **Cố định seed:** đặt `random_state=42` / `torch.manual_seed(42)` ở các bước chia tập, CV và khởi tạo mô hình để kết quả tái lập được.
4. **Biểu đồ rõ ràng, không copy-paste nhầm tiêu đề:** mỗi biểu đồ bắt buộc đúng `title`, `xlabel`, `ylabel`, có số liệu trên cột/heatmap và lưu bản ảnh rõ nét vào `reports/figures/` khi đưa vào báo cáo.
5. **Không lặp code thủ công cho nhiều mô hình:** dùng vòng lặp qua cấu hình hoặc helper khi chạy nhiều biến thể (như E02, E03, E07) để đảm bảo cùng thước đo và không gõ nhầm tham số.

---

## 5. Checklist rà soát trước khi commit / nộp bài

- [ ] Notebook chạy sạch từ đầu tới cuối (**Restart & Run All**) không lỗi thứ tự cell?
- [ ] Có đoạn code nào bị copy-paste giữa nhiều notebook thay vì gọi từ `src/` không?
- [ ] Có chỗ nào vô tình fit TF-IDF/ngưỡng hoặc chọn tham số bằng tập `test` không?
- [ ] Đã kiểm tra `.shape` và trùng lặp giữa `train` / `dev` / `test` sau bước làm sạch chưa?
- [ ] Tiêu đề, nhãn trục của mọi biểu đồ và phần kết luận ngắn đã khớp với số liệu vừa chạy chưa?
- [ ] Đã chạy `python -m pytest tests -q` (nếu có sửa `src/`) trước khi push chưa?
