# Student Career Fuzzy

Hệ thống tư vấn hướng nghiệp và lựa chọn môn tự chọn THPT dựa trên điểm số học sinh. Dự án kết hợp **Fuzzy C-Means (FCM)**, feature engineering và bộ điểm có trọng số để đề xuất Top 2 môn tự chọn phù hợp, sau đó ánh xạ sang các tổ hợp xét tuyển phổ biến.

## Tính năng

- Làm sạch dữ liệu điểm từ file Excel.
- Chuẩn hóa điểm và tạo các đặc trưng năng lực, bao gồm điểm trung bình và xu hướng tiến bộ.
- Phân cụm năng lực bằng Fuzzy C-Means với 3 nhóm: `Tự nhiên`, `Xã hội`, `Ngoại ngữ`.
- Đánh giá kết quả phân cụm bằng Fuzzy Partition Coefficient (FPC) và Fuzzy Partition Entropy (FPE).
- Tính điểm phù hợp và đề xuất Top 2 môn: Vật lý, Hóa học, Sinh học, Lịch sử, Địa lý, Tiếng Anh.
- Ánh xạ cặp môn được đề xuất sang các tổ hợp xét tuyển THPT.
- Tra cứu từng học sinh trên giao diện Streamlit với biểu đồ radar, biểu đồ membership và chức năng tải báo cáo CSV.

## Công nghệ

- Python 3.10+
- Pandas, NumPy, OpenPyXL
- scikit-learn, scikit-fuzzy
- Streamlit, Plotly, Matplotlib, Seaborn

## Cấu trúc dự án

```text
.
├── app.py                         # Giao diện Streamlit
├── src/
│   ├── main.py                    # Pipeline xử lý chính
│   ├── config.py                  # Đường dẫn, môn học, trọng số và ánh xạ tổ hợp
│   ├── data/                      # Đọc và tiền xử lý dữ liệu
│   ├── features/                  # Tạo đặc trưng và chuẩn hóa
│   ├── clustering/                # Fuzzy C-Means và đánh giá mô hình
│   ├── counseling/                # Đề xuất môn và tổ hợp xét tuyển
│   └── visualization/             # Các biểu đồ trong ứng dụng
├── data/
│   ├── raw/                       # Dữ liệu đầu vào
│   └── processed/                 # Dữ liệu trung gian và kết quả
├── results/                       # Hình ảnh, biểu đồ và kết quả phân tích
├── notebooks/                     # Notebook phục vụ phân tích
├── requirements.txt
└── README.md
```

## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Kích hoạt trên macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Đặt file dữ liệu đầu vào tại:

```text
data/raw/student_scores.xlsx
```

File Excel cần giữ cấu trúc mà bộ tiền xử lý đang sử dụng:

- Hai dòng đầu là header.
- Cột thông tin học sinh gồm mã học sinh, họ tên và lớp.
- Điểm các môn được sắp theo thứ tự dự kiến trong file: Toán, Lý, Hóa, Sinh, cột trung gian, Văn, Địa/GDĐP, Sử, Anh.
- Dữ liệu điểm được cung cấp cho lớp 10, lớp 11 và học kỳ 1 lớp 12.

Các giá trị thiếu ở cột điểm sẽ được thay thế bằng median của cột; nếu toàn bộ cột trống, giá trị `0.0` được sử dụng.

## Chạy pipeline

Từ thư mục gốc dự án, chạy:

```bash
python src/main.py
```

Pipeline thực hiện lần lượt:

1. Làm sạch dữ liệu Excel.
2. Tạo đặc trưng và chuẩn hóa Min-Max.
3. Phân cụm Fuzzy C-Means và tính FPC/FPE.
4. Tính điểm và chọn Top 2 môn tự chọn.
5. Ánh xạ sang các tổ hợp xét tuyển.

## Chạy ứng dụng

Sau khi pipeline hoàn tất, khởi động giao diện:

```bash
streamlit run app.py
```

Mở URL được Streamlit hiển thị trong terminal, thường là `http://localhost:8501`. Có thể lọc theo lớp, chọn học sinh, xem các biểu đồ năng lực và tải toàn bộ báo cáo tư vấn dưới dạng CSV.

## Công thức điểm tư vấn

Điểm cuối của mỗi môn tự chọn được tính theo:

```text
Final Score = 0.6 × Subject Score
			+ 0.3 × Cluster Fit
			+ 0.1 × Trend Score
```

Trong đó:

- `Subject Score`: điểm môn đã chuẩn hóa về khoảng `[0, 1]`.
- `Cluster Fit`: mức độ thuộc cụm năng lực tương ứng từ FCM.
- `Trend Score`: xu hướng tiến bộ được đưa qua hàm sigmoid.

## Các file kết quả

Sau khi chạy pipeline, thư mục `data/processed/` sẽ có:

| File | Nội dung |
| --- | --- |
| `cleaned_scores.csv` | Dữ liệu điểm sau tiền xử lý |
| `features.csv` | Đặc trưng năng lực của học sinh |
| `normalized_scores.csv` | Điểm đã chuẩn hóa |
| `membership.csv` | Mức độ thuộc các cụm FCM |
| `top2_recommendations.csv` | Hai môn tự chọn được đề xuất |
| `final_counseling_results.csv` | Kết quả cuối cùng kèm tổ hợp xét tuyển |

## Lưu ý

- Cần chạy `python src/main.py` trước khi chạy Streamlit; ứng dụng đọc các file CSV đã xử lý trong `data/processed/`.
- Nếu đổi vị trí hoặc tên file Excel, hãy cập nhật `RAW_EXCEL_PATH` trong `src/config.py`.
- Kết quả là công cụ tham khảo dựa trên dữ liệu điểm, không thay thế tư vấn chuyên môn hoặc thông tin tuyển sinh chính thức.
