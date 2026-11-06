"""
SMS Shield 4.0 - Design Tokens, Custom CSS & Interactive Plotly Theming Engine
Author: SMS Shield NLP Project

Provides unified SaaS cybersecurity styling tokens, dark-mode CSS injection,
reusable HTML components, and pre-configured Plotly chart builders.
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# =============================================================================
# Cybersecurity SaaS Color Palette Tokens
# =============================================================================
BG_DARK = "#0B1020"            # Deep midnight navy
CARD_BG = "#151E33"            # Elevated dark card
CARD_BG_ELEVATED = "#1B2740"   # Hover & highlighted card
SIDEBAR_BG = "#10172A"         # Secondary dark slate
BORDER_COLOR = "#26334D"       # Subtle border stroke
BORDER_ACCENT = "#3B82F6"      # Subtle blue border
PRIMARY_VIOLET = "#8B5CF6"     # Electric violet
PRIMARY_VIOLET_LIGHT = "#A78BFA"
CYAN_ACCENT = "#38BDF8"        # High-tech cyan
CYAN_GLOW = "rgba(56, 189, 248, 0.25)"
SUCCESS_GREEN = "#10B981"      # Emerald green (Clean / Legitimate)
SUCCESS_GREEN_BG = "rgba(16, 185, 129, 0.12)"
DANGER_RED = "#F87171"         # Soft crimson red (Spam / Threat)
DANGER_RED_BG = "rgba(248, 113, 113, 0.12)"
WARNING_AMBER = "#F59E0B"      # Amber alert
WARNING_AMBER_BG = "rgba(245, 158, 11, 0.12)"
TEXT_PRIMARY = "#F8FAFC"       # High-contrast crisp white
TEXT_MUTED = "#94A3B8"         # Slate gray for descriptions
TEXT_DIM = "#64748B"           # Subdued tertiary text


# =============================================================================
# Custom Cybersecurity SaaS CSS
# =============================================================================
CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #F8FAFC;
    }

    /* Main container breathing room */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1350px;
    }

    /* Dark Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B1020;
    }
    ::-webkit-scrollbar-thumb {
        background: #26334D;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3B82F6;
    }

    /* Hero Header Banner */
    .cyber-hero {
        background: radial-gradient(100% 180% at 0% 0%, #1E1B4B 0%, #111827 50%, #0B1020 100%);
        border: 1px solid #26334D;
        border-top: 2px solid #8B5CF6;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .cyber-hero::after {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 350px;
        height: 100%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .cyber-hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .cyber-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 0;
        background: linear-gradient(135deg, #FFFFFF 20%, #A78BFA 60%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cyber-subtitle {
        font-size: 0.98rem;
        font-weight: 400;
        color: #94A3B8;
        margin-top: 6px;
        line-height: 1.5;
    }
    .cyber-telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
        align-items: center;
    }

    /* Status Pills & Indicators */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 9999px;
        background: #151E33;
        border: 1px solid #26334D;
        color: #E2E8F0;
    }
    .status-pill.active {
        border-color: #38BDF8;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
    }
    .status-pill.violet {
        border-color: #8B5CF6;
        color: #C4B5FD;
        background: rgba(139, 92, 246, 0.12);
    }
    .status-pill.success {
        border-color: #10B981;
        color: #34D399;
        background: rgba(16, 185, 129, 0.12);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.online {
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-green 2s infinite;
    }
    .status-dot.threat {
        background-color: #F87171;
        box-shadow: 0 0 10px #F87171;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 6px rgba(248, 113, 113, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }

    /* Elevated Cyber Cards */
    .cyber-card {
        background: #151E33;
        border: 1px solid #26334D;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease-in-out;
    }
    .cyber-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 6px 20px rgba(11, 16, 32, 0.6);
        transform: translateY(-2px);
    }
    .cyber-card-elevated {
        background: #1B2740;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    /* Executive KPI Metric Cards */
    .cyber-kpi-card {
        background: #151E33;
        border: 1px solid #26334D;
        border-radius: 12px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 105px;
        transition: border-color 0.2s ease;
    }
    .cyber-kpi-card:hover {
        border-color: #8B5CF6;
    }
    .cyber-kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .cyber-kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .cyber-kpi-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: #38BDF8;
        margin-top: 4px;
    }

    /* Threat Verdict Banners */
    .verdict-banner-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(127, 29, 29, 0.22) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.18);
    }
    .verdict-banner-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 78, 59, 0.22) 100%);
        border: 1.5px solid #10B981;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.18);
    }

    /* Badges */
    .verdict-spam {
        background: rgba(239, 68, 68, 0.2);
        border: 1.5px solid #EF4444;
        color: #FCA5A5;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .verdict-ham {
        background: rgba(16, 185, 129, 0.2);
        border: 1.5px solid #10B981;
        color: #6EE7B7;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Monospace / Code styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.88em;
        border: 1px solid #334155;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #10172A;
        border-right: 1px solid #1E293B;
    }

    /* Form Inputs and Text Areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #151E33 !important;
        color: #F8FAFC !important;
        border: 1px solid #26334D !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.3) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B1020;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #26334D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #151E33 !important;
        color: #F8FAFC !important;
        border-bottom: 2px solid #8B5CF6 !important;
    }

    /* Streamlit Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""


def inject_theme_css():
    """Injects the comprehensive cyber SaaS CSS stylesheet into the current Streamlit page."""
    st.markdown(CYBER_CSS, unsafe_allow_html=True)


# =============================================================================
# Plotly Dark Theming Engine
# =============================================================================
def apply_plotly_theme(fig: go.Figure, height: int = 360, title: Optional[str] = None) -> go.Figure:
    """Applies a consistent dark cybersecurity aesthetic to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_PRIMARY, size=12),
        margin=dict(l=40, r=30, t=50 if title else 30, b=40),
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=14, color=TEXT_PRIMARY),
            x=0.02,
            y=0.95
        ) if title else None,
        legend=dict(
            bgcolor="rgba(21, 30, 51, 0.8)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(size=11, color=TEXT_MUTED)
        ),
    )
    fig.update_xaxes(
        gridcolor="#1E293B",
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12)
    )
    fig.update_yaxes(
        gridcolor="#1E293B",
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_PRIMARY, size=12)
    )
    return fig


def create_donut_chart(
    labels: List[str],
    values: List[float],
    colors: List[str],
    title: str = "",
    center_text: str = "",
    height: int = 340
) -> go.Figure:
    """Creates a high-contrast cybersecurity donut chart."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.64,
                marker=dict(colors=colors, line=dict(color=BORDER_COLOR, width=1.5)),
                textinfo="label+percent",
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=12, family="Inter"),
                hoverinfo="label+value+percent",
            )
        ]
    )
    if center_text:
        fig.add_annotation(
            text=f"<b>{center_text}</b>",
            x=0.5,
            y=0.5,
            font_size=16,
            font_color=TEXT_PRIMARY,
            showarrow=False,
        )
    apply_plotly_theme(fig, height=height, title=title)
    fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=40 if title else 20, b=20))
    return fig


def create_comparative_bar_chart(
    df: pd.DataFrame,
    model_col: str,
    metric_cols: List[str],
    colors: Optional[List[str]] = None,
    title: str = "Classifier Performance Across Evaluation Metrics",
    height: int = 380
) -> go.Figure:
    """Creates a grouped bar chart comparing multiple ML algorithms across metrics."""
    if colors is None:
        colors = [PRIMARY_VIOLET, CYAN_ACCENT, SUCCESS_GREEN, "#F59E0B", "#EC4899"]

    fig = go.Figure()
    for idx, metric in enumerate(metric_cols):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=df[model_col],
                y=df[metric] * 100 if df[metric].max() <= 1.0 else df[metric],
                marker_color=colors[idx % len(colors)],
                text=[f"{v*100:.1f}%" if v <= 1.0 else f"{v:.1f}%" for v in df[metric]],
                textposition="outside",
                textfont=dict(color=TEXT_PRIMARY, size=10, family="Inter"),
            )
        )

    apply_plotly_theme(fig, height=height, title=title)
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 115]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
