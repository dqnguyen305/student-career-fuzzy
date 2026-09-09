import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_PROCESSED_DIR

ALL_SUBJECTS = [
    "math", "physics", "chemistry", "biology", "informatics",
    "literature", "geography", "history", "english"
]

def create_features(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Tính điểm trung bình các môn, gom nhóm 3 miền năng lực (Tự nhiên, Xã hội, Ngoại ngữ) 
    và chuẩn hóa [0, 1] cho FCM với trọng số bình đẳng 1:1:1.
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
            
            # Thay thế điểm <= 0 bằng NaN để không kéo tụt điểm trung bình khi thiếu dữ liệu
            sub_scores = df_clean[[col_10, col_11, col_12]].replace(0, np.nan)
            
            # Tính trung bình các kỳ có điểm hợp lệ (> 0)
            features_df[avg_col_name] = sub_scores.mean(axis=1).round(2)
            features_df[avg_col_name] = features_df[avg_col_name].fillna(0.0)
            avg_cols.append(avg_col_name)
            
            # Xu hướng tiến bộ: So sánh HK1 Lớp 12 với trung bình Lớp 10 + 11
            prev_avg = df_clean[[col_10, col_11]].replace(0, np.nan).mean(axis=1)
            features_df[f"{sub}_trend"] = (df_clean[col_12] - prev_avg).fillna(0.0).round(2)

    # 2. BỔ SUNG: Gom nhóm 3 miền năng lực đại diện để tránh Feature Count Imbalance
    features_df["natural_score"] = features_df[[
        "math_avg", "physics_avg", "chemistry_avg", "biology_avg", "informatics_avg"
    ]].mean(axis=1).round(2)
    features_df["social_score"] = features_df[["literature_avg", "history_avg", "geography_avg"]].mean(axis=1).round(2)
    features_df["english_score"] = features_df["english_avg"]

    # 3. Chuẩn hóa về [0, 1], sau đó đưa mỗi học sinh về cùng mặt phẳng
    # tổng bằng 0 để FCM học miền nổi trội thay vì học chênh lệch mức điểm.
    domain_cols = ["natural_score", "social_score", "english_score"]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0, 0, 0], [10, 10, 10]]))
    domain_scaled = scaler.transform(features_df[domain_cols].values)
    relative_scores = domain_scaled - domain_scaled.mean(axis=1, keepdims=True)
    X_scaled_df = pd.DataFrame(relative_scores, columns=["natural", "social", "english"], index=features_df.index)

    # Lưu Scaler ra file để sử dụng đồng bộ tại Tab 2 Streamlit Real-time
    scaler_path = os.path.join(DATA_PROCESSED_DIR, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)

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
    except Exception as err:
        print(f"❌ Lỗi Feature Engineering: {err}")