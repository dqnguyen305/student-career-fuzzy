import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
from src.config import DATA_PROCESSED_DIR, ALL_EXAM_COMBINATIONS, SUBJECT_MAP_VN

# Ánh xạ từ tiền tố tên cột điểm (math, chemistry...) sang Tên Môn Tiếng Việt Chuẩn
SUBJECT_PREFIX_MAP = {
    'math': 'Toán',
    'literature': 'Văn',
    'physics': 'Lý',
    'chemistry': 'Hóa',
    'biology': 'Sinh',
    'history': 'Sử',
    'geography': 'Địa',
    'english': 'Anh'
}

def extract_subject_averages(row: pd.Series) -> dict:
    """
    Gom nhóm tất cả các cột điểm (lớp 10, 11, 12_hk1) của từng môn 
    và tính điểm trung bình môn cho học sinh.
    Ví dụ: [math_10: 6.5, math_11: 7.3, math_12_hk1: 7.7] -> Toán: 7.17
    """
    subject_scores = {vn_name: [] for vn_name in SUBJECT_PREFIX_MAP.values()}

    for col_name, val in row.items():
        clean_col = str(col_name).lower().strip()
        
        # Bỏ qua nếu giá trị trống/NaN
        if pd.isna(val):
            continue
        if pd.isna(val) or (isinstance(val, (int, float, np.number)) and float(val) <= 0):
            continue

        # Kiểm tra tiền tố môn học trong tên cột (ví dụ: chemistry_10 -> chemistry)
        for prefix, vn_name in SUBJECT_PREFIX_MAP.items():
            if clean_col.startswith(prefix):
                try:
                    subject_scores[vn_name].append(float(val))
                except (ValueError, TypeError):
                    pass
                break

    # Tính điểm trung bình môn của học sinh (nếu không có điểm thì mặc định 0.0)
    final_avg = {}
    for vn_name, scores in subject_scores.items():
        if scores:
            final_avg[vn_name] = float(np.mean(scores))
        else:
            final_avg[vn_name] = 0.0

    return final_avg


def map_subjects_to_combinations(top2_df: pd.DataFrame) -> pd.DataFrame:
    mapped_results = top2_df.copy()
    
    clean_scores_path = os.path.join(DATA_PROCESSED_DIR, "cleaned_scores.csv")
    if not os.path.exists(clean_scores_path):
        clean_scores_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")

    scores_df = pd.read_csv(clean_scores_path)

    # Đảm bảo student_id là kiểu chuỗi chuẩn
    if "student_id" in mapped_results.columns:
        mapped_results["student_id"] = mapped_results["student_id"].astype(str).str.strip()
    if "student_id" in scores_df.columns:
        scores_df["student_id"] = scores_df["student_id"].astype(str).str.strip()

    # Tính điểm trung bình các môn cho từng học sinh và lưu vào dict theo student_id
    student_scores_dict = {}
    for idx, row in scores_df.iterrows():
        s_id = str(row.get("student_id", idx)).strip()
        student_scores_dict[s_id] = extract_subject_averages(row)

    combinations_list = []

    for idx, row in mapped_results.iterrows():
        s_id = str(row.get("student_id", idx)).strip()
        
        # Lấy bảng điểm môn của học sinh (nếu lệch ID thì lấy theo dòng index)
        scores_map = student_scores_dict.get(s_id, {})
        if not scores_map and idx in scores_df.index:
            fallback_id = str(scores_df.iloc[idx].get("student_id", idx)).strip()
            scores_map = student_scores_dict.get(fallback_id, {})

        # Tên 2 môn tự chọn
        sub1_eng = str(row.get("top1_subject", "")).lower().strip()
        sub2_eng = str(row.get("top2_subject", "")).lower().strip()
        
        sub1_vn = SUBJECT_MAP_VN.get(sub1_eng, row.get("top1_subject"))
        sub2_vn = SUBJECT_MAP_VN.get(sub2_eng, row.get("top2_subject"))

        student_4_subjects = set(["Toán", "Văn", sub1_vn, sub2_vn])
        valid_combis = []

        for code, subjects in ALL_EXAM_COMBINATIONS.items():
            if set(subjects).issubset(student_4_subjects):
                # Tính tổng điểm 3 môn từ điểm trung bình các năm
                total_score = sum(scores_map.get(sub, 0.0) for sub in subjects)
                subjects_str = f"({', '.join(subjects)})"
                valid_combis.append((code, round(total_score, 2), subjects_str))

        valid_combis.sort(key=lambda x: x[1], reverse=True)
        top4 = valid_combis[:4]
        
        combi_text = " | ".join([f"{code} {subs}: {score:.2f}" for code, score, subs in top4])
        combinations_list.append(combi_text)

    mapped_results["suggested_combinations"] = combinations_list
    return mapped_results