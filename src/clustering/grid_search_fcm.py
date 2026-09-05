import os
import sys
import numpy as np
import pandas as pd
import skfuzzy as fuzz

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import DATA_PROCESSED_DIR
from src.clustering.evaluation import evaluate_fcm

def run_grid_search():
    features_path = os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv")
    if not os.path.exists(features_path):
        print(f"❌ Không tìm thấy {features_path}. Vui lòng chạy main.py trước!")
        return

    X_scaled_df = pd.read_csv(features_path)
    X = X_scaled_df.values  # Shape: (N, features)
    data = X.T              # Shape: (features, N) cho skfuzzy

    n_clusters_list = [2, 3, 4, 5]
    m_list = [1.2, 1.3, 1.5, 1.7, 2.0]
    seeds = [10, 42, 100, 2024]

    results = []

    print("🔍 ĐANG THỰC HIỆN GRID SEARCH & KIỂM TRA ĐỘ ỔN ĐỊNH MÔ HÌNH FCM...\n")

    for c in n_clusters_list:
        for m in m_list:
            seed_metrics = []
            for seed in seeds:
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data, c=c, m=m, error=0.005, maxiter=1000, seed=seed
                )
                metrics = evaluate_fcm(X, cntr, u, m=m)
                seed_metrics.append(metrics)

            # Tính trung bình và độ lệch chuẩn qua các seed để đánh giá độ ổn định
            avg_fpc = np.mean([m_["fpc"] for m_ in seed_metrics])
            std_fpc = np.std([m_["fpc"] for m_ in seed_metrics])
            
            avg_fpe = np.mean([m_["fpe"] for m_ in seed_metrics])
            avg_xb = np.mean([m_["xie_beni"] for m_ in seed_metrics])
            avg_sil = np.mean([m_["silhouette"] for m_ in seed_metrics])
            avg_dbi = np.mean([m_["davies_bouldin"] for m_ in seed_metrics])

            results.append({
                "n_clusters": c,
                "m_fuzziness": m,
                "fpc_mean": avg_fpc,
                "fpc_std": std_fpc,
                "fpe_mean": avg_fpe,
                "xie_beni_mean": avg_xb,
                "silhouette_mean": avg_sil,
                "davies_bouldin_mean": avg_dbi
            })

    results_df = pd.DataFrame(results)
    
    # Sắp xếp kết quả theo Silhouette cao nhất và Xie-Beni thấp nhất
    results_df = results_df.sort_values(by=["silhouette_mean", "xie_beni_mean"], ascending=[False, True])

    # Lưu kết quả ra CSV để đưa vào chương Báo cáo Thực nghiệm
    output_path = os.path.join(DATA_PROCESSED_DIR, "fcm_grid_search_results.csv")
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("✅ Đã hoàn thành Grid Search! Kết quả top 5 cấu hình tốt nhất:\n")
    print(results_df.head(5).to_string(index=False))
    print(f"\n📂 Kết quả đầy đủ đã được lưu tại: {output_path}")

if __name__ == "__main__":
    run_grid_search()