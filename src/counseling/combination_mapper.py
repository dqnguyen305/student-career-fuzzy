import os
import sys

# Tự động thêm thư mục gốc dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.config import DATA_PROCESSED_DIR, COMBINATIONS_MAP, BLOCK_DETAILS

def map_subjects_to_combinations(top2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ánh xạ Top 2 môn tự chọn đề xuất thành các tổ hợp kèm chi tiết tên môn thi.
    """
    mapped_results = top2_df.copy()
    combinations_list = []

    for i in range(len(mapped_results)):
        sub1 = mapped_results.loc[i, "top1_subject"]
        sub2 = mapped_results.loc[i, "top2_subject"]

        # Sắp xếp 2 môn theo thứ tự bảng chữ cái
        pair = tuple(sorted([sub1, sub2]))

        # Lấy danh sách mã khối thi
        matched_blocks = COMBINATIONS_MAP.get(pair, ["D01"])

        # Chuyển đổi từ mã khối sang dạng có chi tiết môn: "A00 (Toán, Lý, Hóa)"
        detailed_blocks = [BLOCK_DETAILS.get(code, code) for code in matched_blocks]

        # Ghép các khối lại bằng dấu phẩy
        combinations_list.append(" | ".join(detailed_blocks))

    mapped_results["suggested_combinations"] = combinations_list
    return mapped_results

if __name__ == "__main__":
    try:
        top2_path = os.path.join(DATA_PROCESSED_DIR, "top2_recommendations.csv")
        if not os.path.exists(top2_path):
            raise FileNotFoundError("❌ Chưa có top2_recommendations.csv, hãy chạy subject_recommender.py trước.")

        top2_df = pd.read_csv(top2_path)
        print("🔄 Đang ánh xạ Top 2 môn sang Tổ hợp xét tuyển chi tiết...")

        final_recommendations = map_subjects_to_combinations(top2_df)

        output_path = os.path.join(DATA_PROCESSED_DIR, "final_counseling_results.csv")
        final_recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ Đã lưu báo cáo tư vấn tổ hợp chi tiết tại: {output_path}")
        print("\n--- Xem 3 kết quả đầu tiên ---")
        print(final_recommendations[["student_name", "top1_subject", "top2_subject", "suggested_combinations"]].head(3))

    except Exception as err:
        print(f"❌ Lỗi Mapper: {err}")