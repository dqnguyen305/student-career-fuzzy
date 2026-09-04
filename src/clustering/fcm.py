import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import skfuzzy as fuzzy
from src.config import DATA_PROCESSED_DIR, CLUSTER_NAMES

def run_fcm_clustering(
    X_scaled_df: pd.DataFrame, 
    n_clusters: int = 3, 
    m: float = 1.5, 
    error: float = 0.005, 
    maxiter: int = 1000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Thực hiện phân cụm mờ Fuzzy C-Means bằng scikit-fuzzy.
    
    Parameters:
        X_scaled_df: DataFrame dữ liệu đã Min-Max Scaling (Học sinh x Môn học).
        n_clusters (C): Số cụm (Mặc định = 3).
        m: Trọng số mờ (Fuzziness exponent, thường = 2.0).
        
    Returns:
        cntr: Trọng tâm các cụm (Centroids) [C x Num_Features].
        u: Ma trận độ thuộc Membership Matrix U [C x Num_Students].
        fpc: Chỉ số Fuzzy Partition Coefficient (Đánh giá độ phân tách cụm).
        cluster_mapping: Ánh xạ từ chỉ số cụm (0, 1, 2) sang tên cụm (Tự nhiên, Xã hội, Ngoại ngữ).
    """
    # scikit-fuzzy yêu cầu ma trận đầu vào dạng (Num_Features x Num_Students)
    data_matrix = X_scaled_df.T.values

    # Chạy Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzzy.cmeans(
        data=data_matrix,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=seed
    )

    # 1. Tự động diễn giải Centroids để gán nhãn Tự nhiên, Xã hội, Ngoại ngữ
    # Feature index: math(0), physics(1), chemistry(2), biology(3), literature(4), geography(5), history(6), english(7)
    cols = list(X_scaled_df.columns)
    
    natural_subs = ["math", "physics", "chemistry", "biology"]
    social_subs = ["literature", "geography", "history"]
    language_subs = ["english"]

    nat_indices = [cols.index(s) for s in natural_subs if s in cols]
    soc_indices = [cols.index(s) for s in social_subs if s in cols]
    lang_indices = [cols.index(s) for s in language_subs if s in cols]

    cluster_mapping = {}
    remaining_clusters = list(range(n_clusters))

    # Cụm Tự nhiên: Có trung bình các môn Tự nhiên cao nhất
    nat_scores = [np.mean(cntr[i, nat_indices]) for i in remaining_clusters]
    best_nat_idx = remaining_clusters[np.argmax(nat_scores)]
    cluster_mapping[best_nat_idx] = "Tự nhiên"
    remaining_clusters.remove(best_nat_idx)

    # Cụm Ngoại ngữ: Có điểm môn Anh cao nhất trong các cụm còn lại
    lang_scores = [np.mean(cntr[i, lang_indices]) for i in remaining_clusters]
    best_lang_idx = remaining_clusters[np.argmax(lang_scores)]
    cluster_mapping[best_lang_idx] = "Ngoại ngữ"
    remaining_clusters.remove(best_lang_idx)

    # Cụm Xã hội: Cụm còn lại
    cluster_mapping[remaining_clusters[0]] = "Xã hội"

    return cntr, u, fpc, cluster_mapping

if __name__ == "__main__":
    try:
        scaled_path = os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv")
        features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
        
        if not os.path.exists(scaled_path):
            raise FileNotFoundError("❌ Chưa có file normalized_scores.csv, hãy chạy feature_engineering.py trước.")

        X_scaled_df = pd.read_csv(scaled_path)
        features_df = pd.read_csv(features_path)

        print("🔄 Đang thực hiện phân cụm mờ Fuzzy C-Means (FCM)...")
        cntr, u, fpc, cluster_mapping = run_fcm_clustering(X_scaled_df, n_clusters=3)

        print(f"✅ Phân cụm hoàn tất! Chỉ số FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
        print("\n--- Ánh xạ Cụm Năng Lực ---")
        for idx, name in cluster_mapping.items():
            print(f"Cluster {idx} ──> Nhóm năng lực: {name}")

        print("\n--- Bảng Trọng Tâm Các Cụm (Centroids) ---")
        cntr_df = pd.DataFrame(cntr, columns=X_scaled_df.columns)
        cntr_df["Cluster_Name"] = [cluster_mapping[i] for i in range(len(cntr))]
        print(cntr_df.set_index("Cluster_Name"))

        # Tạo DataFrame lưu ma trận Membership Matrix U cho từng học sinh
        u_df = pd.DataFrame(u.T, columns=[f"membership_{cluster_mapping[i]}" for i in range(len(cntr))])
        membership_result = pd.concat([features_df[["student_id", "student_name", "class"]], u_df], axis=1)

        # Lưu kết quả Membership Matrix
        output_path = os.path.join(DATA_PROCESSED_DIR, "membership.csv")
        membership_result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Đã lưu ma trận độ thuộc Membership U tại: {output_path}")

        print("\n--- Xem 3 dòng đầu ma trận Membership (Độ thuộc) ---")
        print(membership_result.head(3))

    except Exception as err:
        print(f"❌ Lỗi chạy FCM: {err}")