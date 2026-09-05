import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import pandas as pd
from src.data.preprocessor import preprocess_student_data
from src.features.feature_engineering import create_features
from src.clustering.fcm import (
    run_fcm_clustering,
    generate_cluster_profiles,
    build_cluster_label_diagnostics,
)
from src.clustering.evaluation import evaluate_fcm
from src.counseling.subject_recommender import recommend_top_subjects
from src.counseling.combination_mapper import map_subjects_to_combinations
from src.config import DATA_PROCESSED_DIR, RAW_EXCEL_PATH, FCM_FUZZINESS

def run_pipeline():
    print("=" * 60)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE TƯ VẤN HƯỚNG NGHỆP MỜ (FUZZY CAREER)")
    print("=" * 60)

    # Bước 1: Preprocessing
    print("\n[1/5] 🔄 Đang làm sạch dữ liệu từ Excel...")
    df_clean = preprocess_student_data(RAW_EXCEL_PATH)
    clean_out = os.path.join(DATA_PROCESSED_DIR, "cleaned_scores.csv")
    df_clean.to_csv(clean_out, index=False, encoding="utf-8-sig")

    # Bước 2: Feature Engineering
    print("\n[2/5] 🔄 Đang trích xuất đặc trưng & Chuẩn hóa Min-Max...")
    features_df, X_scaled_df, scaler = create_features(df_clean)
    features_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "features.csv"), index=False, encoding="utf-8-sig")
    X_scaled_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv"), index=False, encoding="utf-8-sig")

    # Bước 3: Fuzzy C-Means Clustering & Evaluation
    print("\n[3/5] 🔄 Đang phân cụm mờ Fuzzy C-Means (FCM) & Đánh giá mô hình...")
    cntr, u, fpc_val, cluster_mapping, raw_u = run_fcm_clustering(
        X_scaled_df, n_clusters=3, m=FCM_FUZZINESS
    )
    
    # Kích hoạt đánh giá FPC và FPE từ module evaluation (truyền đủ X_scaled_df.values, cntr, u)
    eval_metrics = evaluate_fcm(X_scaled_df.values, cntr, raw_u, m=FCM_FUZZINESS)
    print(f"   ├─ Fuzzy Partition Coefficient (FPC): {eval_metrics['fpc']:.4f} (Càng gần 1 càng tốt)")
    print(f"   └─ Fuzzy Partition Entropy (FPE)    : {eval_metrics['fpe_normalized']:.4f} (Càng gần 0 càng tốt)")
    pd.DataFrame([eval_metrics]).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "evaluation_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    build_cluster_label_diagnostics(
        cntr,
        cluster_mapping,
        list(X_scaled_df.columns)
    ).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "cluster_label_diagnostics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------
    # BỔ SUNG: XUẤT FILE CENTROIDS.CSV ĐỂ THAM CHIẾU TẠI TAB 2 STREAMLIT
    # ------------------------------------------------------------------
    cntr_df = pd.DataFrame(cntr, columns=X_scaled_df.columns)
    cntr_df["Cluster_Name"] = [cluster_mapping[i] for i in range(len(cntr))]
    cntr_out = os.path.join(DATA_PROCESSED_DIR, "centroids.csv")
    cntr_df.to_csv(cntr_out, index=False, encoding="utf-8-sig")
    print(f"   ├─ Đã xuất ma trận Tâm cụm (Centroids) tại: {cntr_out}")

    # Lưu Ma trận Membership U
    u_df = pd.DataFrame(u.T, columns=[f"membership_{cluster_mapping[i]}" for i in range(len(cntr))])
    membership_result = pd.concat([features_df[["student_id", "student_name", "class"]], u_df], axis=1)
    membership_result.to_csv(os.path.join(DATA_PROCESSED_DIR, "membership.csv"), index=False, encoding="utf-8-sig")

    profile_df, class_dist_df = generate_cluster_profiles(
        features_df, membership_result, list(features_df.columns)
    )
    profile_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "cluster_profile_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    class_dist_df.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "cluster_class_distribution.csv"),
        encoding="utf-8-sig"
    )
    # Bước 4: Subject Recommender
    print("\n[4/5] 🔄 Đang tính điểm Final Score tư vấn môn tự chọn...")
    top2_df = recommend_top_subjects(features_df, X_scaled_df, membership_result)
    top2_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "top2_recommendations.csv"), index=False, encoding="utf-8-sig")

    # Bước 5: Combination Mapping
    print("\n[5/5] 🔄 Đang ánh xạ sang các Tổ hợp xét tuyển THPT...")
    final_df = map_subjects_to_combinations(top2_df)
    final_out = os.path.join(DATA_PROCESSED_DIR, "final_counseling_results.csv")
    final_df.to_csv(final_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("🎉 PIPELINE HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()