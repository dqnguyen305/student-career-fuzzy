import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_EXCEL_PATH = os.path.join(DATA_RAW_DIR, "student_scores.xlsx")

# Danh sách môn tính điểm chính
CORE_SUBJECTS = ["Toán", "Lý", "Hóa", "Sinh", "Văn", "Địa", "Sử", "Anh"]

# Danh sách môn tự chọn đề xuất (Loại trừ Toán và Văn)
OPTIONAL_SUBJECTS = ["physics", "chemistry", "biology", "history", "geography", "english"]

# Tên nhóm cụm mờ năng lực
CLUSTER_NAMES = {
    0: "Tự nhiên",
    1: "Xã hội",
    2: "Ngoại ngữ"
}

# Trọng số tính Final Score tư vấn môn
WEIGHT_SUBJECT_SCORE = 0.6  # w1: Điểm trung bình môn
WEIGHT_CLUSTER_FIT = 0.3    # w2: Mức độ thuộc cụm (Membership)
WEIGHT_TREND_SCORE = 0.1    # w3: Xu hướng tiến bộ

# Quy tắc ánh xạ 2 môn tự chọn -> Top 2-3 Tổ hợp thi THPT phổ biến nhất
COMBINATIONS_MAP = {
    # --- Môn Tự nhiên & Tự nhiên ---
    ("chemistry", "physics"): ["A00", "A01", "D07"],          # Lý - Hóa -> Toán-Lý-Hóa, Toán-Lý-Anh, Toán-Hóa-Anh
    ("biology", "physics"): ["A02", "B08"],                   # Lý - Sinh -> Toán-Lý-Sinh, Toán-Sinh-Anh
    ("biology", "chemistry"): ["B00", "D07", "B08"],          # Hóa - Sinh -> Toán-Hóa-Sinh, Toán-Hóa-Anh, Toán-Sinh-Anh

    # --- Môn Xã hội & Xã hội ---
    ("geography", "history"): ["C00", "C04", "D14"],          # Sử - Địa -> Văn-Sử-Địa, Văn-Toán-Địa, Văn-Sử-Anh

    # --- Môn Tự nhiên & Xã hội ---
    ("history", "physics"): ["A03", "C07", "D09"],            # Lý - Sử -> Toán-Lý-Sử, Văn-Sử-Lý, Toán-Sử-Anh
    ("geography", "physics"): ["A04", "C04", "D10"],          # Lý - Địa -> Toán-Lý-Địa, Văn-Toán-Địa, Toán-Địa-Anh
    ("chemistry", "history"): ["A05", "C10", "D07"],          # Hóa - Sử -> Toán-Hóa-Sử, Văn-Sử-Hóa, Toán-Hóa-Anh
    ("chemistry", "geography"): ["A06", "C11", "D07"],        # Hóa - Địa -> Toán-Hóa-Địa, Văn-Địa-Hóa, Toán-Hóa-Anh
    ("biology", "history"): ["B01", "B03", "D08"],            # Sinh - Sử -> Toán-Sinh-Sử, Toán-Sinh-Văn, Toán-Sinh-Anh
    ("biology", "geography"): ["B02", "B03", "D08"],          # Sinh - Địa -> Toán-Sinh-Địa, Toán-Sinh-Văn, Toán-Sinh-Anh

    # --- Môn Ngoại ngữ (Anh) & Môn khác ---
    ("english", "physics"): ["A01", "D11"],                   # Lý - Anh -> Toán-Lý-Anh, Văn-Lý-Anh
    ("chemistry", "english"): ["D07", "D12"],                 # Hóa - Anh -> Toán-Hóa-Anh, Văn-Hóa-Anh
    ("biology", "english"): ["B08", "D08", "D13"],            # Sinh - Anh -> Toán-Sinh-Anh, Văn-Sinh-Anh
    ("english", "history"): ["D09", "D14"],                   # Sử - Anh -> Toán-Sử-Anh, Văn-Sử-Anh
    ("english", "geography"): ["D10", "D15"],                 # Địa - Anh -> Toán-Địa-Anh, Văn-Địa-Anh
}
# Bảng giải thích chi tiết các môn học trong từng khối thi
BLOCK_DETAILS = {
    "A00": "A00 (Toán, Lý, Hóa)",
    "A01": "A01 (Toán, Lý, Anh)",
    "A02": "A02 (Toán, Lý, Sinh)",
    "A03": "A03 (Toán, Lý, Sử)",
    "A04": "A04 (Toán, Lý, Địa)",
    "A05": "A05 (Toán, Hóa, Sử)",
    "A06": "A06 (Toán, Hóa, Địa)",
    "A07": "A07 (Toán, Sử, Địa)",
    "B00": "B00 (Toán, Hóa, Sinh)",
    "B01": "B01 (Toán, Sinh, Sử)",
    "B02": "B02 (Toán, Sinh, Địa)",
    "B03": "B03 (Toán, Sinh, Văn)",
    "B08": "B08 (Toán, Sinh, Anh)",
    "C00": "C00 (Văn, Sử, Địa)",
    "C01": "C01 (Văn, Toán, Lý)",
    "C02": "C02 (Văn, Toán, Hóa)",
    "C03": "C03 (Văn, Toán, Sử)",
    "C04": "C04 (Văn, Toán, Địa)",
    "C05": "C05 (Văn, Lý, Hóa)",
    "C06": "C06 (Văn, Lý, Sinh)",
    "C07": "C07 (Văn, Sử, Lý)",
    "C08": "C08 (Văn, Hóa, Sinh)",
    "C09": "C09 (Văn, Địa, Lý)",
    "C10": "C10 (Văn, Sử, Hóa)",
    "C11": "C11 (Văn, Địa, Hóa)",
    "C12": "C12 (Văn, Sử, Sinh)",
    "C13": "C13 (Văn, Sinh, Địa)",
    "D01": "D01 (Toán, Văn, Anh)",
    "D07": "D07 (Toán, Hóa, Anh)",
    "D08": "D08 (Toán, Sinh, Anh)",
    "D09": "D09 (Toán, Sử, Anh)",
    "D10": "D10 (Toán, Địa, Anh)",
    "D11": "D11 (Văn, Lý, Anh)",
    "D12": "D12 (Văn, Hóa, Anh)",
    "D13": "D13 (Văn, Sinh, Anh)",
    "D14": "D14 (Văn, Sử, Anh)",
    "D15": "D15 (Văn, Địa, Anh)",
}