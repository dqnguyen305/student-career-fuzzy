import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from src.config import RAW_EXCEL_PATH, DATA_PROCESSED_DIR

def preprocess_student_data(file_path: str = RAW_EXCEL_PATH) -> pd.DataFrame:
    """
    Tiền xử lý dữ liệu điểm Excel bằng định vị index chính xác cho đủ 8 môn.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file_path}")

    # Đọc bỏ qua 2 dòng header trên cùng
    df_raw = pd.read_excel(file_path, skiprows=2, engine="openpyxl")

    cleaned_df = pd.DataFrame()

    # 1. Trích xuất thông tin học sinh
    cleaned_df["student_id"] = df_raw.iloc[:, 3].astype(str).str.replace(".0", "", regex=False)
    cleaned_df["student_name"] = df_raw.iloc[:, 2]
    cleaned_df["class"] = df_raw.iloc[:, 1]

    # 2. Offset chuẩn xác theo từng môn:
    # 0: Toán, 1: Lý, 2: Hóa, 3: Sinh, 5: Văn, 6: GDĐP, 7: Sử, 8: Anh
    subject_offsets = {
        "math": 0,
        "physics": 1,
        "chemistry": 2,
        "biology": 3,
        "literature": 5,
        "geography": 6,  # Cột Giáo dục địa phương / Địa lý trong file
        "history": 7,
        "english": 8
    }

    # Chỉ số cột bắt đầu môn Toán chính xác cho từng giai đoạn
    periods_start_idx = {
        "10": 4,      # Môn Toán Lớp 10 bắt đầu từ cột index 4
        "11": 17,     # Môn Toán Lớp 11 bắt đầu từ cột index 17
        "12_hk1": 30  # Môn Toán HK1 Lớp 12 bắt đầu từ cột index 30
    }

    # 3. Bóc tách điểm số
    for period_code, start_idx in periods_start_idx.items():
        for sub_key, offset in subject_offsets.items():
            col_idx = start_idx + offset
            if col_idx < df_raw.shape[1]:
                col_name = f"{sub_key}_{period_code}"
                cleaned_df[col_name] = pd.to_numeric(df_raw.iloc[:, col_idx], errors='coerce')

    # 4. Lọc bỏ các dòng không hợp lệ
    cleaned_df = cleaned_df.dropna(subset=["student_name"]).reset_index(drop=True)
    cleaned_df = cleaned_df[~cleaned_df["student_name"].astype(str).str.contains("Tên|Họ|học sinh", case=False)]

    # 5. Xử lý Missing Values an toàn (Chỉ lấy median nếu cột có ít nhất 1 giá trị hợp lệ)
    score_cols = [c for c in cleaned_df.columns if c not in ["student_id", "student_name", "class"]]
    for col in score_cols:
        valid_scores = cleaned_df[col].dropna()
        if len(valid_scores) > 0:
            median_val = valid_scores.median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
        else:
            cleaned_df[col] = cleaned_df[col].fillna(0.0)

    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    try:
        print("🔄 Đang xử lý làm sạch dữ liệu...")
        df_cleaned = preprocess_student_data()
        
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(DATA_PROCESSED_DIR, "cleaned_scores.csv")
        df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print(f"✅ Đã xử lý xong! Dữ liệu sạch lưu tại: {output_path}")
        print(f"\n--- Tổng số cột bóc tách: {len(df_cleaned.columns)} (Kỳ vọng: 27 = 3 info + 24 điểm) ---")
        print("Danh sách các cột điểm:")
        print(df_cleaned.columns.tolist())
        print("\n--- Kiểm tra 3 dòng điểm môn English 11 & History 12 HK1 ---")
        print(df_cleaned[["student_name", "english_11", "history_12_hk1"]].head(3))
    except Exception as err:
        print(f"❌ Lỗi: {err}")