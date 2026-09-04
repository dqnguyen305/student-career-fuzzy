import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_PROCESSED_DIR

ALL_SUBJECTS = ["math", "physics", "chemistry", "biology", "literature", "geography", "history", "english"]

def create_features(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Tính điểm trung bình (xử lý bỏ qua điểm 0 lỗi), xu hướng tiến bộ và chuẩn hóa [0, 1].
    """
    features_df = pd.DataFrame()
    features_df["student_id"] = df_clean["student_id"]
    features_df["student_name"] = df_clean["student_name"]
    features_df["class"] = df_clean["class"]

    avg_cols = []
    
    # 1. Tính điểm trung bình và xu hướng cho từng môn
    for sub in ALL_SUBJECTS:
        col_10 = f"{sub}_10"
        col_11 = f"{sub}_11"
        col_12 = f"{sub}_12_hk1"
        
        if col_10 in df_clean.columns and col_11 in df_clean.columns and col_12 in df_clean.columns:
            avg_col_name = f"{sub}_avg"
            
            # Thay thế điểm <= 0 bằng NaN để không làm kéo tụt điểm trung bình khi bị thiếu dữ liệu kỳ đó
            sub_scores = df_clean[[col_10, col_11, col_12]].replace(0, np.nan)
            
            # Tính trung bình các kỳ có điểm hợp lệ (> 0)
            features_df[avg_col_name] = sub_scores.mean(axis=1).round(2)
            
            # Nếu tất cả các kỳ đều bằng 0, điền mặc định 0.0
            features_df[avg_col_name] = features_df[avg_col_name].fillna(0.0)
            avg_cols.append(avg_col_name)
            
            # Xu hướng tiến bộ: So sánh HK1 Lớp 12 với trung bình Lớp 10 + 11 (bỏ qua điểm 0)
            prev_avg = df_clean[[col_10, col_11]].replace(0, np.nan).mean(axis=1)
            features_df[f"{sub}_trend"] = (df_clean[col_12] - prev_avg).fillna(0.0).round(2)

    # 2. Chuẩn hóa về thang [0, 1] cho FCM bằng cách chia 10 (giữ nguyên bản chất điểm gốc 0-10)
    # Không dùng MinMaxScaler vì nó làm bóp méo khoảng cách điểm thực tế giữa các môn
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Lấy dữ liệu điểm trung bình
    X_raw = features_df[avg_cols].values
    
    # Fit scaler trên khung cố định điểm từ 0 đến 10
    scaler.fit(np.array([[0]*len(avg_cols), [10]*len(avg_cols)]))
    X_scaled_array = scaler.transform(X_raw)
    
    scaled_col_names = [col.replace("_avg", "") for col in avg_cols]
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=scaled_col_names, index=features_df.index)

    return features_df, X_scaled_df, scaler

if __name__ == "__main__":
    try:
        clean_path = os.path.join(DATA_PROCESSED_DIR, "cleaned_scores.csv")
        if not os.path.exists(clean_path):
            raise FileNotFoundError("❌ Chưa có file cleaned_scores.csv, hãy chạy preprocessor.py trước.")

        df_clean = pd.read_csv(clean_path)
        print("🔄 Đang thực hiện Feature Engineering...")
        
        features_df, X_scaled_df, scaler = create_features(df_clean)

        features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
        scaled_path = os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv")
        
        features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
        X_scaled_df.to_csv(scaled_path, index=False, encoding="utf-8-sig")

        print(f"✅ Đã lưu tập đặc trưng đầy đủ tại: {features_path}")
        print(f"✅ Đã lưu tập dữ liệu chuẩn hóa [0, 1] tại: {scaled_path}")
        
        print("\n--- Xem 3 dòng đầu của điểm trung bình môn (features.csv) ---")
        print(features_df[["student_name", "english_avg", "physics_avg", "chemistry_avg"]].head(3))
    except Exception as err:
        print(f"❌ Lỗi Feature Engineering: {err}")