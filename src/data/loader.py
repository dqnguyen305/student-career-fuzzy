import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path để tránh lỗi ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.config import RAW_EXCEL_PATH

def load_student_data(file_path: str = RAW_EXCEL_PATH, sheet_name: str | int = 0) -> pd.DataFrame:
    """
    Nạp dữ liệu thô điểm học sinh từ file Excel (.xlsx).
    Giữ nguyên cấu trúc MultiHeader (2 dòng) để preprocessor xử lý.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {file_path}")
        
    try:
        # Đọc dữ liệu Excel giữ nguyên 2 dòng header
        df = pd.read_excel(file_path, header=[0, 1], engine="openpyxl")
        print(f"✅ Đã nạp thành công {len(df)} bản ghi từ file Excel: {file_path}")
        return df
    except Exception as e:
        raise RuntimeError(f"❌ Lỗi khi đọc file Excel: {str(e)}")

if __name__ == "__main__":
    try:
        df_raw = load_student_data()
        print("\n--- Xem thử 3 dòng dữ liệu thô ---")
        print(df_raw.head(3))
    except Exception as err:
        print(err)