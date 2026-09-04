import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from src.config import DATA_PROCESSED_DIR
from src.visualization.radar import plot_student_radar
from src.visualization.membership_chart import plot_membership_bar

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ Thống Tư Vấn Môn Học Mờ (Fuzzy Career)",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
    membership_path = os.path.join(DATA_PROCESSED_DIR, "membership.csv")
    final_path = os.path.join(DATA_PROCESSED_DIR, "final_counseling_results.csv")
    
    features_df = pd.read_csv(features_path)
    membership_df = pd.read_csv(membership_path)
    final_df = pd.read_csv(final_path)

    # Đảm bảo student_id luôn là chuỗi để so sánh chính xác 100%
    features_df["student_id"] = features_df["student_id"].astype(str)
    membership_df["student_id"] = membership_df["student_id"].astype(str)
    final_df["student_id"] = final_df["student_id"].astype(str)

    return features_df, membership_df, final_df

try:
    features_df, membership_df, final_df = load_data()
except Exception as e:
    st.error("❌ Chưa tìm thấy dữ liệu đã xử lý. Hãy chạy `python src/main.py` trước!")
    st.stop()

# Header ứng dụng
st.title("🎓 Hệ Thống Tư Vấn Hướng Nghiệp & Chọn Môn Thi THPT (Fuzzy Logic)")
st.markdown("Phân tích năng lực học sinh dựa trên **Fuzzy C-Means Clustering** và thuật toán tư vấn có trọng số.")

# Sidebar bộ lọc
st.sidebar.header("🔍 Lọc & Tra Cứu Học Sinh")
selected_class = st.sidebar.selectbox("Chọn Lớp:", options=["Tất cả"] + list(final_df["class"].dropna().unique()))

if selected_class != "Tất cả":
    filtered_df = final_df[final_df["class"] == selected_class]
else:
    filtered_df = final_df

# Tối ưu: Định danh kết hợp [Tên - Mã HS] tránh trùng lặp
filtered_df["display_name"] = filtered_df["student_name"] + " (ID: " + filtered_df["student_id"] + ")"
student_dict = dict(zip(filtered_df["display_name"], filtered_df["student_id"]))

selected_display = st.sidebar.selectbox("Chọn Học Sinh:", options=list(student_dict.keys()))
selected_s_id = student_dict[selected_display]

# Trích xuất chính xác dữ liệu của 1 học sinh theo student_id
student_info = final_df[final_df["student_id"] == selected_s_id].iloc[0]
s_features = features_df[features_df["student_id"] == selected_s_id].iloc[0]
s_membership = membership_df[membership_df["student_id"] == selected_s_id].iloc[0]

# Hiển thị thông tin tổng quan
st.subheader(f"📌 Kết Quả Tư Vấn: {student_info['student_name']} (Lớp {student_info['class']} - Mã HS: {selected_s_id})")

col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.metric(
        label="🎯 Môn Tự Chọn Ưu Tiên 1", 
        value=f"{str(student_info['top1_subject']).upper()}", 
        delta=f"Score: {float(student_info['top1_score']):.4f}"
    )

with col2:
    st.metric(
        label="🎯 Môn Tự Chọn Ưu Tiên 2", 
        value=f"{str(student_info['top2_subject']).upper()}", 
        delta=f"Score: {float(student_info['top2_score']):.4f}"
    )

with col3:
    st.markdown("**📚 Tổ Hợp Xét Tuyển Đề Xuất:**")
    combis = [c.strip() for c in str(student_info['suggested_combinations']).split('|')]
    for combi in combis:
        st.markdown(f"🔹 `{combi}`")

st.divider()

# Biểu đồ trực quan hóa
col_left, col_right = st.columns(2)

with col_left:
    st.write("### 🕸️ Biểu Đồ Radar Năng Lực Các Môn")
    fig_radar = plot_student_radar(s_features, student_info['student_name'])
    st.plotly_chart(fig_radar, use_container_width=True)

with col_right:
    st.write("### 📊 Mức Độ Thuộc Cụm Mờ Năng Lực (FCM Membership)")
    fig_bar = plot_membership_bar(s_membership)
    st.plotly_chart(fig_bar, use_container_width=True)

# Bảng dữ liệu tổng thể và Xuất CSV
with st.expander("📄 Xem Danh Sách Báo Cáo Tư Vấn Toàn Bộ Học Sinh"):
    # Bỏ cột hỗ trợ hiển thị trước khi render bảng
    display_df = final_df.drop(columns=["display_name"], errors="ignore")
    st.dataframe(display_df, use_container_width=True)
    
    csv_data = display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Tải Báo Cáo Kết Quả (CSV)",
        data=csv_data,
        file_name="baocao_tuvanhuongnghiep.csv",
        mime="text/csv"
    )