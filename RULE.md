# RULE.md — Quy tắc viết code & chạy thực nghiệm (Local / Colab / Kaggle)

Mọi code mới hoặc chỉnh sửa trong `notebooks/`, `experiments/`, `src/` bắt buộc tuân theo các quy tắc dưới đây.

---

## 1. Tương thích 3 môi trường: Local, Google Colab, Kaggle GPU

Không viết code chỉ chạy được trên máy Windows local hoặc chỉ chạy được trên Colab. Mọi notebook (đặc biệt các file nặng dùng GPU như `04_phobert_multitask.ipynb`, `E04`, `E05`) phải tự nhận diện môi trường ở cell đầu tiên:

1. **Phát hiện môi trường:**
   ```python
   IN_COLAB = 'google.colab' in sys.modules
   IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working')
   ```
2. **Định vị `ROOT` (đọc `src/`, `data/`) và `WORK_DIR` (nơi được phép ghi `models/`, `reports/`, `.tmp_run/`):**
   - **Local:** `ROOT = os.path.abspath('..')` nếu `cwd` là `notebooks` hoặc `experiments`, ngược lại `os.getcwd()`. `WORK_DIR = ROOT`.
   - **Google Colab:** tìm ở `.` hoặc `/content`; nếu chưa có `src/` mà có `/content/src_and_data.zip` thì tự giải nén. `WORK_DIR = ROOT`.
   - **Kaggle:** `/kaggle/input/...` là **read-only**, `/kaggle/working/` là **writable**.
     - Nếu gắn dataset dạng zip (`src_and_data.zip`) trong `/kaggle/input/...`, giải nén vào `/kaggle/working/project` và đặt `ROOT = WORK_DIR = '/kaggle/working/project'`.
     - Nếu gắn thư mục repo trực tiếp trong `/kaggle/input/<dataset>`, đọc từ `ROOT = '/kaggle/input/<dataset>'` nhưng **mọi thao tác ghi** (`reports/`, `models/`, `.tmp_run/`, `vncorenlp`) bắt buộc trỏ sang `WORK_DIR = '/kaggle/working'`.
     - Nếu dùng `src.exp_log`, set `os.environ['EXP_LOG_PATH'] = os.path.join(WORK_DIR, 'reports', 'experiments_log.csv')` trước khi `import`.
3. **Đường dẫn & OS:** luôn dùng `os.path.join(...)` hoặc `pathlib.Path`, tuyệt đối không hardcode `D:\...`, `/content/...` hay dấu `\`.
4. **Thư viện thiếu:** chỉ `pip install -q` khi `import` thất bại (tránh cài lại ở Local). Với `py_vncorenlp` trên Kaggle/Colab, lưu model vào thư mục ghi được (`os.path.join(WORK_DIR, '.tmp_run', 'vncorenlp')`).
5. **GPU / CPU an toàn:** luôn dùng `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Không gọi `.cuda()` cứng, `pin_memory=(device.type == 'cuda')`.

---

## 2. Quy tắc pipeline & thực nghiệm

- **Pipeline chính (`notebooks/00..06`):** giữ gọn, luôn là bản đã chốt. Không tạo `v2`, `v3` trong `notebooks/`.
- **Thử nghiệm (`experiments/E0x_*.ipynb`):** mỗi file chỉ kiểm tra 1 câu hỏi, đọc từ `data/processed/*_clean.csv` và dùng hàm trong `src/`, ghi kết quả vào `reports/experiments_log.csv`.
- **Chống data leakage:**
  - Mọi quyết định (tuning, chọn model, early stopping, chỉnh ngưỡng) chỉ dùng `dev`, **tuyệt đối không dùng `test`** để chọn cấu hình.
  - Chỉ chạy `test` 1 lần ở notebook chính sau khi đã chốt mô hình.
- **Chạy lại sau khi sửa:** chỉ cần chạy từ bước đã sửa trở về sau; trước khi nộp bài mới chạy sạch `00 -> 06` một lần duy nhất.
- **Phong cách code:** tuân theo [coding_style.md](coding_style.md) — cell ngắn, thử trên mẫu nhỏ trước khi chạy toàn bộ, comment ngắn đúng trọng tâm.
