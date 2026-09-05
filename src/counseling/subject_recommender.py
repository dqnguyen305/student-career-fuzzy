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

# Ánh xạ môn tự chọn với từ khóa đại diện cho Nhóm cụm năng lực tương ứng
SUBJECT_TO_DOMAIN_KEYWORD = {
    "physics": "Tự nhiên",
    "chemistry": "Tự nhiên",
    "biology": "Tự nhiên",
    "history": "Xã hội",
    "geography": "Xã hội",
    "english": "Ngoại ngữ"
}

DOMAIN_SCORE_COLUMNS = {
    "Tự nhiên": "natural_score",
    "Xã hội": "social_score",
    "Ngoại ngữ": "english_score"
}

def normalize_trend_score(trend_val: float, min_val: float = -2.0, max_val: float = 2.0) -> float:
    """
    Chuẩn hóa xu hướng tiến bộ trend_val về khoảng [0, 1] theo Linear Min-Max Clipping.
    """
    clipped = np.clip(trend_val, min_val, max_val)
    return float((clipped - min_val) / (max_val - min_val))


def find_membership_col(membership_cols: list[str], keyword: str) -> str | None:
    """
    Tìm tên cột membership phù hợp nhất chứa từ khóa domain (ví dụ: 'Tự nhiên', 'Xã hội', 'Ngoại ngữ').
    """
    for col in membership_cols:
        if keyword.lower() in col.lower():
            return col
    return None


def recommend_top_subjects(
    features_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    membership_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Tính FinalScore cho các môn tự chọn và đề xuất Top 2 môn phù hợp nhất cho từng học sinh.
    """
    # 1. Khắc phục triệt để trùng lặp cột & index
    df_feat = features_df.loc[:, ~features_df.columns.duplicated()].reset_index(drop=True)
    df_mem = membership_df.loc[:, ~membership_df.columns.duplicated()].reset_index(drop=True)

    recommendations = []
    membership_cols = [c for c in df_mem.columns if c.startswith("membership_")]

    for i in range(len(df_feat)):
        # Lấy thông tin học sinh an toàn bằng .iat
        student_id = str(df_feat["student_id"].iat[i])
        student_name = str(df_feat["student_name"].iat[i])
        student_class = str(df_feat["class"].iat[i])

        subject_scores = {}

        for sub in OPTIONAL_SUBJECTS:
            # 1. Subject Score: Lấy điểm TB môn từ features_df
            avg_col = f"{sub}_avg"
            if avg_col in df_feat.columns:
                val = df_feat[avg_col].iat[i]
                avg_score_scaled = float(np.ravel(val)[0]) / 10.0
            else:
                avg_score_scaled = 0.0

            # 2. Cluster Fit: Lấy độ thuộc cụm từ membership_df dùng .iat để đảm bảo trả về float duy nhất
            domain_kw = SUBJECT_TO_DOMAIN_KEYWORD.get(sub, "")
            cluster_col = find_membership_col(membership_cols, domain_kw)
            
            if cluster_col and cluster_col in df_mem.columns:
                raw_mem = df_mem[cluster_col].iat[i]
                # Sử dụng np.ravel để ép phẳng dữ liệu nếu lỡ bị trả về dạng mảng/Series
                cluster_fit = float(np.ravel(raw_mem)[0])
            else:
                domain_score_col = DOMAIN_SCORE_COLUMNS.get(domain_kw)
                if domain_score_col and domain_score_col in df_feat.columns:
                    cluster_fit = float(df_feat[domain_score_col].iat[i]) / 10.0
                else:
                    cluster_fit = 0.0

            # 3. Trend Score: Chuẩn hóa điểm xu hướng tiến bộ
            trend_col = f"{sub}_trend"
            if trend_col in df_feat.columns:
                raw_trend = df_feat[trend_col].iat[i]
                trend_val = float(np.ravel(raw_trend)[0])
            else:
                trend_val = 0.0
            trend_score = normalize_trend_score(trend_val)

            # 4. Tính Final Score
            final_score = (
                float(WEIGHT_SUBJECT_SCORE) * avg_score_scaled +
                float(WEIGHT_CLUSTER_FIT) * cluster_fit +
                float(WEIGHT_TREND_SCORE) * trend_score
            )
            subject_scores[sub] = final_score

        # Sắp xếp chọn Top 2 môn có điểm cao nhất
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

        if not os.path.exists(features_path) or not os.path.exists(membership_path):
            raise FileNotFoundError("❌ Thiếu file features.csv hoặc membership.csv. Hãy chạy fcm.py trước!")

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