import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from src.config import (
    DATA_PROCESSED_DIR,
    OPTIONAL_SUBJECTS,
    WEIGHT_SUBJECT_SCORE,
    WEIGHT_CLUSTER_FIT,
    WEIGHT_TREND_SCORE
)

# Ánh xạ môn tự chọn với Nhóm cụm năng lực tương ứng
SUBJECT_TO_CLUSTER_MAP = {
    "physics": "membership_Tự nhiên",
    "chemistry": "membership_Tự nhiên",
    "biology": "membership_Tự nhiên",
    "history": "membership_Xã hội",
    "geography": "membership_Xã hội",
    "english": "membership_Ngoại ngữ"
}

def recommend_top_subjects(
    features_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    membership_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Tính FinalScore cho các môn tự chọn (trừ Toán và Văn) và đề xuất Top 2 môn.
    """
    recommendations = []

    for i in range(len(features_df)):
        student_id = features_df.loc[i, "student_id"]
        student_name = features_df.loc[i, "student_name"]
        student_class = features_df.loc[i, "class"]

        subject_scores = {}

        for sub in OPTIONAL_SUBJECTS:
            # 1. Subject Score (Điểm trung bình môn đã chuẩn hóa [0, 1])
            avg_score_scaled = scaled_df.loc[i, sub] if sub in scaled_df.columns else 0.0

            # 2. Cluster Fit (Mức độ thuộc cụm mờ tương ứng)
            cluster_col = SUBJECT_TO_CLUSTER_MAP.get(sub)
            cluster_fit = membership_df.loc[i, cluster_col] if cluster_col in membership_df.columns else 0.0

            # 3. Trend Score (Tính xu hướng tiến bộ chuẩn hóa)
            trend_val = features_df.loc[i, f"{sub}_trend"] if f"{sub}_trend" in features_df.columns else 0.0
            # Biến đổi trend [-10, 10] về khoảng [0, 1] bằng Sigmoid hoặc Min-Max đơn giản
            trend_score = 1.0 / (1.0 + np.exp(-trend_val))

            # 4. Tính Final Score có trọng số
            final_score = (
                WEIGHT_SUBJECT_SCORE * avg_score_scaled +
                WEIGHT_CLUSTER_FIT * cluster_fit +
                WEIGHT_TREND_SCORE * trend_score
            )
            subject_scores[sub] = final_score

        # Sắp xếp chọn Top 2 môn
        sorted_subs = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
        top1_sub, top1_score = sorted_subs[0]
        top2_sub, top2_score = sorted_subs[1]

        recommendations.append({
            "student_id": student_id,
            "student_name": student_name,
            "class": student_class,
            "top1_subject": top1_sub,
            "top1_score": round(top1_score, 4),
            "top2_subject": top2_sub,
            "top2_score": round(top2_score, 4)
        })

    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    try:
        features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
        scaled_path = os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv")
        membership_path = os.path.join(DATA_PROCESSED_DIR, "membership.csv")

        features_df = pd.read_csv(features_path)
        scaled_df = pd.read_csv(scaled_path)
        membership_df = pd.read_csv(membership_path)

        print("🔄 Đang tính toán điểm phù hợp và tư vấn Top 2 môn tự chọn...")
        rec_df = recommend_top_subjects(features_df, scaled_df, membership_df)

        output_path = os.path.join(DATA_PROCESSED_DIR, "top2_recommendations.csv")
        rec_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ Tư vấn hoàn tất! Đã lưu kết quả Top 2 môn tại: {output_path}")
        print("\n--- Xem 5 học sinh đầu tiên ---")
        print(rec_df.head(5))

    except Exception as err:
        print(f"❌ Lỗi Recommender: {err}")