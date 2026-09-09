import pandas as pd
import plotly.graph_objects as go

def plot_student_radar(student_features: pd.Series, student_name: str = "") -> go.Figure:
    """
    Tạo biểu đồ Radar hiển thị điểm trung bình 9 môn của học sinh.
    """
    categories = [
        "Toán", "Lý", "Hóa", "Sinh", "Tin học",
        "Văn", "Địa", "Sử", "Anh"
    ]
    
    scores = [
        student_features.get("math_avg", 0),
        student_features.get("physics_avg", 0),
        student_features.get("chemistry_avg", 0),
        student_features.get("biology_avg", 0),
        student_features.get("informatics_avg", 0),
        student_features.get("literature_avg", 0),
        student_features.get("geography_avg", 0),
        student_features.get("history_avg", 0),
        student_features.get("english_avg", 0)
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=student_name,
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 10],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    return fig