# Student Career Fuzzy

Hệ thống phân tích năng lực học sinh và tư vấn môn thi THPT dựa trên điểm số. Dự án sử dụng tiền xử lý dữ liệu, feature engineering, Fuzzy C-Means (FCM), luật chấm điểm môn học và ánh xạ tổ hợp xét tuyển.

Ứng dụng cung cấp hai cách sử dụng:

- **Tra cứu theo danh sách lớp:** xem kết quả đã tính cho từng học sinh.
- **Nhập điểm trực tiếp:** nhập điểm ba học kỳ và xem membership FCM theo thời gian thực.

## 1. Mục tiêu nghiệp vụ

Hệ thống hỗ trợ trả lời ba câu hỏi:

1. Học sinh có xu hướng mạnh hơn ở miền nào: **Tự nhiên**, **Xã hội** hay **Ngoại ngữ**?
2. Học sinh có mức độ giao thoa giữa nhiều miền hay không?
3. Hai môn tự chọn và các tổ hợp xét tuyển nào phù hợp với hồ sơ điểm hiện tại?

FCM không gán mỗi học sinh vào đúng một nhóm cứng. Mỗi học sinh có một vector membership gồm ba giá trị trong khoảng `[0, 1]`, và tổng ba giá trị bằng `1`.

Ví dụ:

```text
Nhóm Tự nhiên : 0.20
Nhóm Xã hội   : 0.35
Nhóm Ngoại ngữ: 0.45
```

Hồ sơ này nghiêng về Ngoại ngữ nhưng vẫn có giao thoa đáng kể với Xã hội.

## 2. Kiến trúc xử lý

Pipeline chính nằm trong `src/main.py` và gồm năm bước:

```text
Excel đầu vào
    |
    v
Tiền xử lý và xử lý giá trị thiếu
    |
    v
Tính điểm trung bình, xu hướng và điểm 3 miền
    |
    v
Chuẩn hóa, FCM và đánh giá mô hình
    |
    v
Membership, hồ sơ cụm và đề xuất môn
    |
    v
Tổ hợp xét tuyển + giao diện Streamlit
```

### 2.1. Tiền xử lý

Module: `src/data/preprocessor.py`

Hệ thống đọc file Excel bằng `openpyxl`, bỏ hai dòng header đầu và trích xuất:

- Mã học sinh.
- Họ tên.
- Lớp.
- Điểm Toán, Lý, Hóa, Sinh, Văn, Địa, Sử, Anh ở ba giai đoạn:
  - Lớp 10.
  - Lớp 11.
  - Học kỳ 1 lớp 12.

Giá trị thiếu được thay bằng median của chính cột điểm. Nếu cả cột không có giá trị hợp lệ, hệ thống dùng `0.0`.

Các điểm `0` khi tính trung bình môn được xem là dữ liệu thiếu để không kéo giảm điểm do chưa có dữ liệu.

### 2.2. Feature engineering

Module: `src/features/feature_engineering.py`

Với mỗi môn, hệ thống tính:

```text
subject_avg = mean(điểm lớp 10, điểm lớp 11, điểm lớp 12 HK1)
subject_trend = điểm lớp 12 HK1 - mean(điểm lớp 10, điểm lớp 11)
```

Ba miền năng lực được tính như sau:

```text
natural_score  = mean(Toán, Lý, Hóa, Sinh)
social_score   = mean(Văn, Sử, Địa)
english_score  = Anh
```

Các điểm miền được chia cho `10` để đưa về `[0, 1]`. Sau đó, hệ thống trừ trung bình ba miền của từng học sinh để tập trung vào **miền nổi trội tương đối**:

```text
relative_domain = scaled_domain - mean(natural, social, english)
```

Do đó, vector ba miền có tổng gần bằng `0`. Đây là chủ ý thiết kế: học sinh có điểm tuyệt đối cao ở cả ba miền vẫn được xem là cân bằng nếu không miền nào nổi trội hơn.

File kết quả là `data/processed/normalized_scores.csv`.

## 3. Mô hình Fuzzy C-Means

Module: `src/clustering/fcm.py`

### 3.1. Huấn luyện FCM

FCM được chạy với:

- Số cụm: `3`.
- Hệ số mờ: `m = 2.5`, khai báo tại `src/config.py`.
- Sai số hội tụ: `0.005`.
- Số vòng lặp tối đa: `1000`.
- Seed: `42`.

FCM gốc tự học các centroid từ dữ liệu. Ma trận membership raw do `scikit-fuzzy` trả về được dùng cho FPC, FPE, Xie-Beni và các chỉ số phân cụm.

### 3.2. Nhãn nghiệp vụ

Ba nhãn hiển thị cố định là:

- `Nhóm Tự nhiên`.
- `Nhóm Xã hội`.
- `Nhóm Ngoại ngữ`.

FCM bản chất là không giám sát nên thứ tự cluster `0, 1, 2` không có ý nghĩa nghiệp vụ. Code ánh xạ các centroid sang ba nhãn dựa trên miền trội tương đối, sau đó bảo đảm mỗi nhãn được sử dụng một lần.

File `cluster_label_diagnostics.csv` ghi lại:

- Chỉ số cluster gốc.
- Nhãn được gán.
- Miền có giá trị centroid cao nhất.
- Độ chênh so với miền đứng thứ hai.
- Tọa độ centroid ở ba chiều.

Nếu `Dominance_Margin` nhỏ, nhãn cụm cần được diễn giải thận trọng vì centroid không có miền trội mạnh.

### 3.3. Membership dùng cho tư vấn

Để nhãn nghiệp vụ không bị sai do centroid học có độ lớn khác nhau, giao diện và file membership sử dụng ba prototype hướng miền:

```text
Tự nhiên  = ( 2, -1, -1)
Xã hội    = (-1,  2, -1)
Ngoại ngữ = (-1, -1,  2)
```

Các vector được chuẩn hóa độ dài trước khi đo khoảng cách theo hướng. Vì vậy:

- Điểm ba miền bằng nhau cho membership gần `33.33%` mỗi nhóm.
- Tăng điểm Anh làm membership Ngoại ngữ tăng.
- Hai nhóm còn lại giảm dần thay vì bị gán cứng vào `0%` ngay lập tức.
- Tổng membership luôn bằng `1`.

Membership được làm mềm bằng softmax trên điểm miền tương đối với `MEMBERSHIP_SCORE_SCALE = 4.0`.

Trong giao diện, ngoài biểu đồ membership còn có:

- Membership cao nhất.
- Membership đứng thứ hai.
- Mức độ giao thoa: `1 - membership_lớn_nhất`.
- Entropy chuẩn hóa của phân bố membership.
- Khoảng cách theo hướng tới từng prototype.

## 4. Đánh giá mô hình

Các chỉ số trong `evaluation_metrics.csv` được tính trên **membership raw của FCM**, không phải membership prototype dùng cho lớp tư vấn. Điều này giữ đúng ý nghĩa đánh giá thuật toán FCM.

### Fuzzy Partition Coefficient

```text
FPC = sum(u_ij^2) / n
```

Với `c` cụm, FPC nằm trong khoảng `[1/c, 1]`. Với ba cụm:

- Gần `1`: phân cụm rõ, membership tập trung.
- Gần `1/3`: phân cụm rất mờ hoặc các cụm khó tách.

### Fuzzy Partition Entropy

```text
FPE = -sum(u_ij * log(u_ij)) / n
```

FPE raw nằm trong `[0, ln(c)]`. Code cũng xuất:

```text
fpe_normalized = fpe / ln(c)
```

Giá trị chuẩn hóa nằm trong `[0, 1]` và dễ đọc hơn:

- Gần `0`: membership ít mờ.
- Gần `1`: membership phân tán đều giữa các cụm.

Các chỉ số bổ sung:

- **Xie-Beni:** đánh giá độ chặt trong cụm và khoảng cách giữa centroid; càng thấp thường càng tốt.
- **Silhouette:** mức phù hợp của điểm với cụm được gán cứng; càng cao càng tốt.
- **Davies-Bouldin:** độ tương đồng giữa các cụm; càng thấp càng tốt.
- **Calinski-Harabasz:** tỷ lệ phân tán giữa cụm và trong cụm; thường càng cao càng tốt.

Không nên dùng một chỉ số duy nhất để kết luận mô hình tốt hay xấu. Cần xem đồng thời chỉ số, centroid, hồ sơ cụm và các ca kiểm thử nghiệp vụ.

## 5. Tư vấn môn học

Module: `src/counseling/subject_recommender.py`

Các môn tự chọn được xét:

- Vật lý.
- Hóa học.
- Sinh học.
- Lịch sử.
- Địa lý.
- Tiếng Anh.

Điểm cuối mỗi môn được tính:

```text
Final Score =
    0.6 × Subject Score
  + 0.3 × Cluster Fit
  + 0.1 × Trend Score
```

Trong đó:

- `Subject Score`: điểm trung bình môn chia cho `10`.
- `Cluster Fit`: membership của miền tương ứng. Nếu dữ liệu không có cột membership đúng tên miền, hệ thống dùng điểm miền chuẩn hóa làm fallback.
- `Trend Score`: xu hướng được clip trong khoảng `[-2, 2]` rồi đưa về `[0, 1]`.

Kết quả được sắp xếp giảm dần và lấy hai môn đầu tiên.

## 6. Ánh xạ tổ hợp xét tuyển

Module: `src/counseling/combination_mapper.py`

Hệ thống lấy Toán, Văn và hai môn tự chọn được đề xuất để tìm các tổ hợp phù hợp trong danh mục cấu hình tại `src/config.py`.

Với mỗi tổ hợp hợp lệ, hệ thống tính tổng điểm trung bình các môn và sắp xếp giảm dần. Kết quả cuối cùng chứa tối đa bốn tổ hợp đề xuất hàng đầu.

Đây là gợi ý kỹ thuật dựa trên điểm số, không thay thế điều kiện tuyển sinh chính thức của từng trường đại học.

## 7. Cấu trúc thư mục

```text
.
├── app.py                              # Giao diện Streamlit
├── requirements.txt                    # Dependency Python
├── README.md                           # Tài liệu dự án
├── src/
│   ├── main.py                         # Pipeline 5 bước
│   ├── config.py                       # Cấu hình, trọng số và tổ hợp
│   ├── data/
│   │   └── preprocessor.py             # Đọc và làm sạch Excel
│   ├── features/
│   │   └── feature_engineering.py      # Tạo đặc trưng và chuẩn hóa
│   ├── clustering/
│   │   ├── fcm.py                      # Huấn luyện FCM và membership
│   │   └── evaluation.py               # FPC, FPE và chỉ số đánh giá
│   ├── counseling/
│   │   ├── subject_recommender.py      # Chọn Top 2 môn
│   │   └── combination_mapper.py       # Ánh xạ tổ hợp
│   └── visualization/
│       ├── radar.py                     # Radar năng lực
│       └── membership_chart.py          # Biểu đồ membership
├── data/
│   ├── raw/                             # Excel đầu vào
│   └── processed/                       # CSV sinh bởi pipeline
├── results/                             # Hình ảnh và kết quả phân tích
└── notebooks/                           # Notebook nghiên cứu
```

## 8. Cài đặt trên Windows

Mở PowerShell tại thư mục gốc dự án:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy trực tiếp executable trong `venv` mà không cần activate.

## 9. Chuẩn bị dữ liệu đầu vào

Đặt file dữ liệu tại:

```text
data/raw/student_scores.xlsx
```

File Excel hiện tại cần giữ cấu trúc cột mà `preprocessor.py` đang đọc:

- Hai dòng đầu là header.
- Thông tin học sinh nằm ở các cột tên, lớp và mã học sinh theo cấu trúc hiện tại.
- Mỗi giai đoạn có các cột môn theo thứ tự Toán, Lý, Hóa, Sinh, Tin học, Văn, Địa, Sử, Anh.
- Có đủ ba giai đoạn: lớp 10, lớp 11 và lớp 12 học kỳ 1.

Nếu thay đổi bố cục Excel, cần cập nhật `subject_offsets` và `periods_start_idx` trong `src/data/preprocessor.py`.

## 10. Chạy pipeline

Từ thư mục gốc:

```powershell
.\venv\Scripts\python.exe src\main.py
```

Hoặc nếu đã activate môi trường:

```powershell
python src/main.py
```

Pipeline sẽ tạo hoặc cập nhật các file trong `data/processed/`.

## 11. Chạy ứng dụng

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ Streamlit hiển thị trong terminal, thường là:

```text
http://localhost:8501
```

Nếu đã chạy pipeline sau khi mở ứng dụng, hãy tải lại trang hoặc khởi động lại Streamlit để giao diện đọc các CSV mới nhất.

## 12. Các file đầu ra

| File | Nội dung |
| --- | --- |
| `cleaned_scores.csv` | Điểm sau tiền xử lý và xử lý missing values |
| `features.csv` | Điểm trung bình, trend và điểm ba miền |
| `normalized_scores.csv` | Vector đặc trưng tương đối đưa vào FCM |
| `centroids.csv` | Tọa độ centroid và nhãn ba nhóm |
| `membership.csv` | Membership dùng cho tư vấn và tra cứu |
| `cluster_label_diagnostics.csv` | Đối soát nhãn và miền trội của centroid |
| `cluster_profile_summary.csv` | Thống kê môn học theo nhóm |
| `cluster_class_distribution.csv` | Phân bố nhóm theo lớp |
| `evaluation_metrics.csv` | FPC, FPE, Xie-Beni và chỉ số bổ sung |
| `top2_recommendations.csv` | Hai môn tự chọn được đề xuất |
| `final_counseling_results.csv` | Kết quả cuối cùng kèm tổ hợp xét tuyển |
| `minmax_scaler.pkl` | Scaler được lưu để tái sử dụng |

## 13. Kiểm thử nhanh

Kiểm tra cú pháp các module chính:

```powershell
python -m py_compile app.py src\clustering\fcm.py src\clustering\evaluation.py src\main.py
```

Kiểm tra membership có tổng bằng `1`:

```powershell
.\venv\Scripts\python.exe -c "import pandas as pd, numpy as np; m=pd.read_csv('data/processed/membership.csv'); c=[x for x in m.columns if x.startswith('membership_')]; print(np.max(np.abs(m[c].sum(axis=1)-1)))"
```

Kiểm thử nghiệp vụ nên bao gồm:

- Ba miền điểm bằng nhau: membership gần `33.33%` mỗi nhóm.
- Tăng riêng điểm Anh: membership Ngoại ngữ tăng dần.
- Tăng riêng nhóm Toán-Lý-Hóa-Sinh: membership Tự nhiên tăng dần.
- Tăng riêng nhóm Văn-Sử-Địa: membership Xã hội tăng dần.
- Hồ sơ lai giữa hai miền: membership của hai nhóm cùng ở mức đáng kể.

## 14. Giới hạn và diễn giải kết quả

- FCM là mô hình không giám sát; nhãn cụm phụ thuộc dữ liệu và cách tạo đặc trưng.
- Membership là độ thuộc mờ, không phải xác suất đỗ đại học hay xác suất thống kê.
- Prototype nghiệp vụ giúp nhãn nhất quán nhưng không thay thế việc kiểm định trên dữ liệu thực tế.
- Nếu dữ liệu thiếu nhiều học sinh Xã hội nổi trội, cụm Xã hội sẽ kém ổn định dù hệ thống vẫn phải hiển thị đủ ba nhóm theo yêu cầu nghiệp vụ.
- Điểm đề xuất và tổ hợp chỉ mang tính tham khảo; cần đối chiếu quy chế tuyển sinh hiện hành.

## 15. Cấu hình quan trọng

Các tham số chính nằm trong `src/config.py`:

```python
FCM_FUZZINESS = 2.5
MEMBERSHIP_SCORE_SCALE = 4.0
WEIGHT_SUBJECT_SCORE = 0.6
WEIGHT_CLUSTER_FIT = 0.3
WEIGHT_TREND_SCORE = 0.1
```

Khi thay đổi tham số FCM hoặc trọng số tư vấn, cần chạy lại `src/main.py` để cập nhật toàn bộ file đầu ra trước khi xem trên Streamlit.

## 16. Lưu ý sử dụng

Luôn chạy pipeline sau khi thay file Excel hoặc thay đổi logic feature engineering. Không chỉnh thủ công `centroids.csv` hoặc `membership.csv`, vì các file này sẽ bị ghi đè ở lần chạy tiếp theo. Kết quả là công cụ hỗ trợ phân tích và tư vấn, không thay thế đánh giá của giáo viên, chuyên gia hướng nghiệp hoặc thông tin tuyển sinh chính thức.
