"""
SMS Shield 4.0 - Design Tokens, Minimalist CSS & Plotly Theming Engine
Author: SMS Shield NLP Project

Provides a refined, modern SaaS design system inspired by Linear, Vercel, and Raycast.
Features subtle dark surfaces, restrained accents, clean typography, and muted charts.
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# =============================================================================
# Refined Minimalist SaaS Color System
# =============================================================================
BG_DARK = "#0F1117"            # Base background
SIDEBAR_BG = "#141720"         # Sidebar background
CARD_BG = "#191E29"            # Standard card surface
ELEVATED_SURFACE = "#202635"   # Elevated card / hover state
BORDER_COLOR = "#292F3D"       # Subtle border stroke
BORDER_HOVER = "#3B4459"       # Muted border highlight on hover

PRIMARY_ACCENT = "#8B7CF6"     # Soft iris / purple for active states
SECONDARY_ACCENT = "#60A5FA"   # Soft clean blue
SUCCESS_GREEN = "#34D399"      # Clean muted emerald (Legitimate / Ham)
DANGER_RED = "#F87171"         # Clean soft red (Threat / Spam)
WARNING_AMBER = "#FBBF24"      # Muted amber
TEXT_PRIMARY = "#F1F5F9"       # Crisp primary text
TEXT_SECONDARY = "#9CA3AF"     # Clean secondary / muted text
TEXT_MUTED = "#6B7280"         # Subdued caption text


# =============================================================================
# Minimalist SaaS Custom CSS
# =============================================================================
SAAS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #F1F5F9;
        -webkit-font-smoothing: antialiased;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
        margin: 0 auto;
    }

    /* Minimal Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0F1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #292F3D;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3B4459;
    }

    /* Refined Page Header */
    .saas-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #292F3D;
    }
    .saas-header-title {
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #F1F5F9;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .saas-header-subtitle {
        font-size: 0.92rem;
        color: #9CA3AF;
        margin-top: 6px;
        line-height: 1.45;
    }

    /* Standardized Card Container */
    .saas-card {
        background: #191E29;
        border: 1px solid #292F3D;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: border-color 0.15s ease;
    }
    .saas-card:hover {
        border-color: #3B4459;
    }

    /* Clean Metric KPI Cards */
    .saas-metric-card {
        background: #191E29;
        border: 1px solid #292F3D;
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 90px;
        transition: border-color 0.15s ease;
    }
    .saas-metric-card:hover {
        border-color: #3B4459;
    }
    .saas-metric-label {
        font-size: 0.74rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9CA3AF;
        margin-bottom: 2px;
    }
    .saas-metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #F1F5F9;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .saas-metric-sub {
        font-size: 0.78rem;
        color: #9CA3AF;
        margin-top: 4px;
    }

    /* Elegant Result Banners */
    .verdict-box-spam {
        background: rgba(248, 113, 113, 0.06);
        border: 1px solid rgba(248, 113, 113, 0.3);
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }
    .verdict-box-ham {
        background: rgba(52, 211, 153, 0.06);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    /* Minimal Badges & Pills */
    .saas-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 4px;
        background: #202635;
        border: 1px solid #292F3D;
        color: #9CA3AF;
    }
    .saas-pill-spam {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: #F87171;
    }
    .saas-pill-ham {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.3);
        color: #34D399;
    }
    .saas-pill-purple {
        background: rgba(139, 124, 246, 0.1);
        border: 1px solid rgba(139, 124, 246, 0.3);
        color: #C4B5FD;
    }

    /* Monospace / Code Elements */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #141720 !important;
        color: #60A5FA !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85em;
        border: 1px solid #292F3D;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #141720;
        border-right: 1px solid #292F3D;
    }

    /* Form Inputs and Text Areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #141720 !important;
        color: #F1F5F9 !important;
        border: 1px solid #292F3D !important;
        border-radius: 6px !important;
        font-size: 0.92rem !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #8B7CF6 !important;
        box-shadow: none !important;
    }

    /* Minimal Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid #292F3D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        color: #9CA3AF;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 8px 4px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #F1F5F9 !important;
        border-bottom: 2px solid #8B7CF6 !important;
    }

    /* Streamlit Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease;
        border: 1px solid #292F3D;
    }
    .stButton>button:hover {
        border-color: #3B4459;
    }

    /* Streamlit Progress Bars */
    .stProgress > div > div > div > div {
        background-color: #8B7CF6;
        border-radius: 4px;
    }
</style>
"""


def inject_theme_css():
    """Injects the refined minimalist SaaS CSS stylesheet into the current Streamlit page."""
    st.markdown(SAAS_CSS, unsafe_allow_html=True)


# =============================================================================
# Plotly Muted Minimalist Theming Engine
# =============================================================================
def apply_plotly_theme(fig: go.Figure, height: int = 320, title: Optional[str] = None) -> go.Figure:
    """Applies a restrained, clean dark aesthetic to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_SECONDARY, size=11),
        margin=dict(l=35, r=20, t=40 if title else 20, b=35),
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=13, color=TEXT_PRIMARY),
            x=0.01,
            y=0.96
        ) if title else None,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_SECONDARY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    fig.update_xaxes(
        gridcolor="#232936",
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, size=10),
        title_font=dict(color=TEXT_PRIMARY, size=11)
    )
    fig.update_yaxes(
        gridcolor="#232936",
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, size=10),
        title_font=dict(color=TEXT_PRIMARY, size=11)
    )
    return fig


def create_donut_chart(
    labels: List[str],
    values: List[float],
    colors: List[str],
    title: str = "",
    center_text: str = "",
    height: int = 300
) -> go.Figure:
    """Creates a clean, minimal donut chart with muted colors."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color=CARD_BG, width=2)),
                textinfo="label+percent",
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=11, family="Inter"),
                hoverinfo="label+value+percent",
            )
        ]
    )
    if center_text:
        fig.add_annotation(
            text=f"<b>{center_text}</b>",
            x=0.5,
            y=0.5,
            font_size=15,
            font_color=TEXT_PRIMARY,
            showarrow=False,
        )
    apply_plotly_theme(fig, height=height, title=title)
    fig.update_layout(showlegend=False, margin=dict(l=15, r=15, t=30 if title else 15, b=15))
    return fig


def create_comparative_bar_chart(
    df: pd.DataFrame,
    model_col: str,
    metric_cols: List[str],
    colors: Optional[List[str]] = None,
    title: str = "Classifier Performance Across Evaluation Metrics",
    height: int = 340
) -> go.Figure:
    """Creates a clean grouped bar chart comparing multiple ML algorithms across metrics."""
    if colors is None:
        colors = [PRIMARY_ACCENT, SECONDARY_ACCENT, SUCCESS_GREEN, WARNING_AMBER, "#A78BFA"]

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
                textfont=dict(color=TEXT_SECONDARY, size=9, family="Inter"),
            )
        )

    apply_plotly_theme(fig, height=height, title=title)
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 115]),
        xaxis=dict(title=""),
    )
    return fig
