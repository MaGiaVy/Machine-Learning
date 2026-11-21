Pipeline nhìn chung sạch: TF-IDF nằm trong Pipeline + CV, có dedup train/dev/test, test chỉ chạm ở cuối. Còn mấy chỗ đáng sửa, xếp theo mức quan trọng:



Quan trọng



Chưa phải ABSA thật. NB00 gộp nhãn thành 1 sentiment cho cả câu (vừa khen vừa chê → “trung lập”), còn aspect chỉ là có/không. Câu “màn hình đẹp nhưng pin tệ” chỉ ra “trung lập” + “có nhắc màn hình, pin”, không biết pin tiêu cực. Data UIT-ViSFD có sẵn kiểu BATTERY#Negative, nên có thể làm 5 aspect × 4 trạng thái (không nhắc / tiêu cực / trung lập / tích cực). Việc gộp này cũng là lý do lớp Trung lập chỉ F1 ≈ 0.50.

Aspect F1 0.906 bị thổi phồng bởi tần suất nhãn. DESIGN\_GEN có mặt ở \~80% câu: đoán toàn “có” cũng được F1 ≈ 0.89, model được 0.892, tức head này gần như không học thêm gì. Hit-any 95.9% cũng vậy, luôn đoán DESIGN\_GEN đã \~80%. Nên báo cáo kèm baseline “đoán toàn 1”.

Primary aspect F1 chỉ \~0.5, mà NB05 chia cụm dựa vào nó. Nhãn aspect\_group sinh từ rule (câu có đủ 5 aspect vẫn ra group 2), còn dự đoán lấy argmax score nên lệch rule. Ví dụ NB03: pred\_asp\_0=1 nhưng pred\_aspect=4, true là 0. Bảng actionable\_insights đang chia theo nhãn chỉ đúng khoảng một nửa. Sửa bằng cách áp đúng rule tạo nhãn lên vector dự đoán, hoặc cho 1 review vào nhiều cụm theo multi-label.

Regex camera/màn hình (NB05) dễ bắt nhầm. cam khớp “màu cam” và “cam kết”, còn nét, mờ, ảnh là từ của màn hình, quay khớp “quay lại”. Kết quả 113 camera vs 16 màn hình hơi lệch, mình nghi over-match.

matched\_keywords là chuỗi gõ cứng, recommended\_action cũng là template cố định. Tên cột dễ làm người dùng app tưởng là kết quả tự rút ra. Ngoài ra SHAP giải thích LR baseline, còn bảng insight dùng dự đoán PhoBERT.

NB04 không lưu output nào, nên mình không kiểm được PhoBERT có hơn baseline (test F1 0.7345) thật không. Bạn gửi model\_metadata.json hoặc chạy lại và giữ output nhé.



Vừa/nhỏ



“Blackbox” bị dùng để chọn model (NB03) và early stopping (NB04), nên thực chất là validation. Số unbiased là Test.

NB06 gọi predict\_proba trên pipeline thắng, mà thắng được chọn tự động. Chạy lại mà LinearSVC thắng thì lỗi. NB06 cũng đang dùng baseline, không phải PhoBERT.

“10/10 PASS” ở NB06 chấm trên 10 file tự tạo (cứng len(tab)==25), detector chỉnh theo chính chúng. Nên thử thêm vài file export thật.

SHAP chỉ lấy X\_te\[:400] (400 dòng đầu, không random) và ngưỡng count ≥ 3, nên “mau”, “hối\_hận” (count 3-4) rất nhiễu. Nên lấy ngẫu nhiên và nâng ngưỡng.

Phân loại lỗi NB03: “kém” bị xếp vào “phủ định” dù là từ tiêu cực, còn “không” có ở hầu hết review. 38% chưa có nghĩa nếu không so với base rate trên câu đoán đúng.

torch.manual\_seed(42) đặt sau khi tạo model nên head chưa cố định seed. Comment “Blackbox (562)” cũ, thực tế 560.

Pyvi: output NB06 có Mà\_e (ghép từ sai), có thể thử VnCoreNLP so sánh như mình nói ở trên

