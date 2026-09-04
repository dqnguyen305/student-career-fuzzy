import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_membership_bar(student_membership: pd.Series) -> go.Figure:
    """
    Tạo biểu đồ cột thể hiện mức độ thuộc của học sinh vào các cụm mờ năng lực.
    """
    clusters = ["Tự nhiên", "Xã hội", "Ngoại ngữ"]
    
    memberships = [
        student_membership.get("membership_Tự nhiên", 0),
        student_membership.get("membership_Xã hội", 0),
        student_membership.get("membership_Ngoại ngữ", 0)
    ]

    fig = px.bar(
        x=clusters,
        y=memberships,
        labels={'x': 'Nhóm Cụm Năng Lực', 'y': 'Độ Thuộc (Membership U)'},
        color=clusters,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    return fig 