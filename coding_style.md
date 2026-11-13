# Hướng dẫn phong cách viết code (rút ra từ notebook của tác giả)

File này mô tả cách tác giả viết notebook ML (bài toán tabular regression) để:
- Bạn tự viết code theo kiểu "người thật làm bài", không bị bóng bẩy kiểu AI.
- Đưa cho AI làm rule: "viết/sửa code theo đúng style trong file này".

> Nguyên tắc chung: **làm từng bước nhỏ, nhìn thấy kết quả rồi mới đi tiếp**, ghi chú ngắn đủ để nhớ mình đang làm gì. Không cố làm cho code "đẹp" hay "hoàn chỉnh".

---

## 1. Cấu trúc notebook

Chia pipeline thành nhiều notebook theo giai đoạn, mỗi notebook một việc:

| Notebook | Việc làm |
| :--- | :--- |
| `0` | Nạp và ghép các bảng thô, lưu ra 1 file |
| `1` | Làm sạch dữ liệu (sửa chuỗi, ép kiểu, điền khuyết), lưu file `_clean` |
| `2` | EDA: thống kê, biểu đồ, groupby |
| `3` | Feature engineering, encoding, chia train/test, scale, lưu file |
| `baseline` | Mô hình đơn giản nhất (Linear Regression) làm mốc |
| `modeling` | Các mô hình phức tạp dần: Decision Tree, Random Forest, XGBoost, rồi tuning |

- Giữa các notebook trao đổi bằng **file CSV** trong `data/raw` và `data/processed`, notebook sau đọc lại file của notebook trước.
- Cell đầu tiên luôn là import thư viện, cell thứ hai là load dữ liệu.
- Mô hình đi theo thứ tự **đơn giản → phức tạp** và so sánh với baseline.

## 2. Cell nhỏ, mỗi cell một việc

- Một cell thường chỉ **1 đến 5 dòng**: một phép biến đổi, hoặc một lệnh xem kết quả.
- Gọi lệnh xem dữ liệu ở **cell riêng**, ngay sau khi vừa biến đổi: `df.info()`, `df.shape`, `df.head()`, `df['col'].value_counts()`.
- Cell cuối đôi khi để trống, hoặc có cell chỉ để "thử" rồi bỏ lại. Không dọn quá kỹ.
- Import thêm thư viện **ngay trước chỗ cần dùng** (ví dụ `from sklearn.tree import DecisionTreeRegressor` ở đầu mục Decision Tree) chứ không gom hết lên đầu.

Ví dụ nhịp làm việc tự nhiên:

```python
# cell 1
df['Weatherconditions'].value_counts()
# cell 2
tempt = "conditions Fog"
# cell 3
tempt.split(" ")
# cell 4 (sau khi thử ổn mới áp dụng cho cả cột)
df['Weatherconditions'] = df['Weatherconditions'].apply(lambda tempt: tempt.split(" ")[1] if pd.notnull(tempt) else tempt)
```

Tức là **thử trên 1 giá trị mẫu trước, rồi mới `apply` cả cột**.

## 3. Cách đặt tên

- Tên biến ngắn, kiểu viết tắt quen tay: `df`, `X`, `y`, `lr`, `dt`, `rf`, `xgb`.
- Hậu tố thể hiện trạng thái: `X_train_scaled`, `X_test_scaled`, `rf_best`, `xgb_best`, `X_train_exp`.
- Biến tạm đặt tên tuỳ hứng, không cần "chuẩn": `tempt`, `hour`, `mse`, `rmse`, `r2` (dùng lại tên cũ cho nhiều mô hình).
- Tên kết quả có hậu tố mô hình: `rmse_test`, `rmse_train`, `r2_rf_tuned`, `adj_r2_xgb_test`.
- Giữ nguyên tên cột gốc của dataset, kể cả tên viết sai (`Time_Orderd`, `Weatherconditions`).
- Tên cột mới đặt kiểu snake_case đơn giản: `distance_km`, `order_hour`, `order_period`.

## 4. Comment

- Viết **ngắn, một dòng**, thường bắt đầu bằng `#` dính liền chữ (`#Test performance`), **không có khoảng trắng** sau `#` và thường **không viết hoa chuẩn, không chấm câu**.
- Mô tả **việc đang làm** hoặc **lý do**, không mô tả thứ hiển nhiên:
  - `#Fit the scaler on TRAIN ONLY, then apply it to both - never fit on test data`
  - `#Sample 25 random combination instead of testing every combination exhaustively`
- Đánh số bước khi có chuỗi thao tác liên tiếp: `# 1. Copy ...`, `#2. Create ...`, `#3 One-hot ...`, `#4 Create a New scaler ...`.
- Có sai chính tả nhỏ và không sửa (`expandede`, `Orderd`, `combination`). Đây là dấu vết người thật.
- Chỉ comment ở chỗ **có quyết định**: chọn median hay mode, drop hay fill, ordinal hay one-hot, fit scaler trên train hay test.
- Không comment kiểu giải thích từng dòng, không docstring dài, không type hints.

Chỉ dùng docstring/comment dài ở **hàm tự viết hiếm hoi** (ví dụ hàm Haversine), và cũng chỉ 1 đoạn comment đầu hàm.

## 5. Markdown trong notebook

- Chỉ dùng để **đánh dấu mốc lớn**, tiêu đề cực ngắn, viết HOA: `MODEL 2: DECISION TREE`, `ENCODING`, `TRAIN/TEST SPLIT`, `FEATURE SCALING`.
- Đôi khi có một câu giải thích lý do, hoặc liệt kê các bước sắp làm (1 đến 5).
- Không có đoạn văn dài, không emoji, không bảng tổng kết, không "Conclusion" cầu kỳ.

## 6. Cách làm sạch và điền khuyết

- Làm theo thứ tự: sửa định dạng chuỗi → ép kiểu số → xử lý giá trị bất hợp lý → điền khuyết.
- Giá trị khuyết số: điền **median**. Phân loại: điền **mode**.
- Dùng vòng `for col in [...]:` để áp cùng một thao tác cho nhiều cột.
- Giá trị phi thực tế: `clip` (rating > 5) hoặc **loại bỏ** hàng (distance > 100 km) khi chiếm tỉ lệ nhỏ.
- Sau mỗi bước quan trọng gọi `df.info()` để kiểm tra non-null.
- Lỗi nhỏ có thể xảy ra và được giữ lại (ví dụ `inplace=True` cùng phép gán), vì tác giả thử và sửa dần.

## 7. EDA

- Lặp lại một mẫu: `value_counts()` → `groupby(...)['target'].agg(['count','mean','median','std'])`.
- Biểu đồ đơn giản: `sns.histplot`, `sns.lineplot`, `sns.heatmap(corr, annot=True)`, `plot(kind='bar')`.
- Mỗi biểu đồ có `plt.title`, `plt.xlabel`, `plt.ylabel` rõ ràng, `plt.show()` ở cuối.
- Copy cell biểu đồ rồi đổi cột, **không đổi title** nên có chỗ title bị lặp (đây là dấu vết copy-paste thật).
- Không kết luận dài bằng chữ, nhìn biểu đồ xong đi tiếp.

## 8. Mô hình và đánh giá

- Cố định `random_state=42` ở mọi chỗ (split, model, search).
- Mỗi mô hình đi cùng nhịp:
  1. Khởi tạo, `fit` trên train.
  2. `predict` trên test, tính `rmse` và `r2`, `print` bằng hai dòng.
  3. Tính lại trên **train** để kiểm tra overfitting.
- In kết quả bằng `print("tên: ", giá trị)`, không format đẹp, không `f-string` dài.
- Tuning bằng `RandomizedSearchCV` (`n_iter` 25 đến 40, `cv=3`, `scoring='r2'`), in `best_params_`, lấy `best_estimator_`.
- Có thể thử ý tưởng mới (thêm feature `order_period`) rồi **so sánh với bản trước** ngay trong cùng notebook.
- Chống data leakage: tạo feature **trước khi scale**, scaler **chỉ `fit` trên train**.

## 9. Điều NÊN bắt chước (tóm tắt)

- [ ] Cell ngắn, xem kết quả liên tục bằng `.info()`, `.head()`, `.value_counts()`.
- [ ] Thử trên 1 giá trị mẫu rồi mới áp dụng cho cả cột.
- [ ] Comment ngắn, giải thích **lý do**, không viết hoa chuẩn, vài lỗi chính tả nhỏ.
- [ ] Markdown tiêu đề HOA ngắn gọn.
- [ ] Tên biến viết tắt (`df`, `rf`, `xgb`, `tempt`).
- [ ] Copy-paste rồi chỉnh nhẹ giữa các mô hình (khối đánh giá lặp lại).
- [ ] Đi từ baseline đơn giản đến mô hình phức tạp.

## 10. Dấu hiệu làm code "trông như AI viết" cần tránh

| Kiểu AI hay viết | Kiểu của tác giả (nên làm) |
| :--- | :--- |
| Gói mọi thứ vào hàm `def evaluate_model(model, X, y, name):` dùng chung | Viết lại 3 dòng `mse/rmse/r2/print` cho từng mô hình |
| Một cell dài làm cả pipeline (load → clean → train → plot) | Mỗi cell 1 việc, 1 đến 5 dòng |
| Comment dài, đủ dấu câu, giải thích mọi dòng | `#Test performance`, chỉ ghi chỗ có quyết định |
| Docstring đầy đủ, type hints, `logging`, `if __name__ == "__main__"` | Không có, chỉ code chạy thẳng |
| Dùng `Pipeline`/`ColumnTransformer` bọc hết preprocessing | Làm tay từng bước: `map`, `get_dummies`, `StandardScaler` |
| Markdown dài: "## Overview", bảng kết quả, emoji, "Key Findings" | Chỉ tiêu đề HOA ngắn |
| Tên biến đẹp và nhất quán tuyệt đối (`evaluation_metrics_dict`) | Tên ngắn, hơi tuỳ hứng (`tempt`, `rmse`, `r2` dùng lại) |
| Mọi thứ sạch, không cell thừa, không lỗi | Có cell trống cuối file, cell thử nghiệm, title biểu đồ copy lặp |
| Ngay từ đầu dùng kỹ thuật cao (Optuna, SHAP, stacking) | Từ từ: Linear → Tree → RF → XGB → RandomizedSearchCV |
| Print nhiều thông tin định dạng đẹp | `print("rf rmse: ", rmse)` |

### Ví dụ trước/sau

**Kiểu AI (tránh):**
```python
def evaluate(model, X_tr, X_te, y_tr, y_te, name):
    """Evaluate a regression model and print RMSE and R2 for train and test sets."""
    for split, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        print(f"{name} {split}: RMSE={rmse:.3f}, R2={r2_score(y, pred):.3f}")
```

**Kiểu tác giả (nên làm), viết tiếng Việt:**
```python
#Hiệu suất trên tập test
mse = mean_squared_error(y_test, rf.predict(X_test_scaled))
rmse = np.sqrt(mse)
r2 = r2_score(y_test, rf.predict(X_test_scaled))
print("rf rmse: ", rmse)
print("rf r2: ", r2)
```

## 11. Quy ước khi viết bằng tiếng Việt

- Comment và markdown viết **tiếng Việt**, vẫn giữ nhịp ngắn như trên: `#Điền khuyết bằng median vì ít bị ảnh hưởng bởi outlier`.
- Tên biến, tên cột, tên hàm thư viện vẫn giữ tiếng Anh.
- Tiêu đề markdown viết HOA không dấu hoặc có dấu đều được, nhưng ngắn: `MÔ HÌNH 2: DECISION TREE`, `MÃ HOÁ`, `CHIA TRAIN/TEST`.
- Không cần sửa hết lỗi gõ nhỏ. Cứ viết tự nhiên như đang ghi chú cho chính mình.

## 12. Checklist rà soát trước khi nộp

- [ ] Có cell nào dài hơn ~10 dòng không? Nếu có, tách ra.
- [ ] Có hàm bọc chung (`def ...`) mà tác giả sẽ không viết không? Gỡ ra viết thẳng.
- [ ] Comment có quá dài, quá "chuẩn sách" không? Rút còn 1 dòng.
- [ ] Mỗi quyết định quan trọng (drop/fill, encode, scale) có 1 dòng comment lý do chưa?
- [ ] Sau mỗi bước làm sạch có cell xem lại (`info`, `head`, `value_counts`) chưa?
- [ ] Mô hình có đi từ baseline lên phức tạp, có so train vs test chưa?
- [ ] Markdown chỉ còn tiêu đề ngắn, không có đoạn văn dài, emoji, bảng tổng kết?
- [ ] `random_state=42` có ở mọi chỗ cần thiết chưa?
- [ ] Scaler có chỉ `fit` trên train không?
