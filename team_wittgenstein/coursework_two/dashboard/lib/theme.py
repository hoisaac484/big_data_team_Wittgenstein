"""Centralised colour palette, fonts, and Plotly template.

Every chart in the dashboard imports COLORS, FACTORS, and PLOTLY_TEMPLATE
from here. Changing a colour in one place updates the whole dashboard.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Colour palette (financial-industry dark theme)
# ---------------------------------------------------------------------------

COLORS = {
    "background": "#0f1419",  # Trading 212 dark
    "surface": "#1a1f2e",  # Card / panel background
    "surface_alt": "#222838",  # Hover / nested surface
    "surface_hover": "#2a3142",
    "border": "#2d3548",
    "border_strong": "#3d4458",
    "text": "#ffffff",  # Primary text
    "text_secondary": "#cbd5e1",  # Sub-labels - clearly readable
    "text_muted": "#94a3b8",  # Tertiary muted
    "primary": "#00c805",  # Trading 212 brand green
    "secondary": "#00b4d8",  # Cyan - benchmark
    "long": "#00c805",  # Bright green - long / positive
    "short": "#ff3b30",  # Bright red - short / negative
    "neutral": "#94a3b8",
    "success": "#00c805",
    "warning": "#ffb020",
    "danger": "#ff3b30",
}

# Factor-specific colours - used in IC weights, factor analysis, deep dive
FACTORS = {
    "value": "#00b4d8",  # Cyan
    "quality": "#ffb020",  # Amber
    "momentum": "#00c805",  # Green
    "low_vol": "#a855f7",  # Purple
}

# Sector palette (11 GICS sectors)
SECTORS = [
    "#3b82f6",
    "#f59e0b",
    "#10b981",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
    "#6366f1",
    "#84cc16",
    "#06b6d4",
]

# ---------------------------------------------------------------------------
# Plotly template - applied globally
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color=COLORS["text"],
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        colorway=[
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["long"],
            COLORS["short"],
            "#a855f7",
            "#ffb020",
            "#06b6d4",
        ],
        xaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border_strong"],
            tickfont=dict(color=COLORS["text_secondary"], size=12),
            title=dict(font=dict(color=COLORS["text_secondary"], size=12)),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border_strong"],
            tickfont=dict(color=COLORS["text_secondary"], size=12),
            title=dict(font=dict(color=COLORS["text_secondary"], size=12)),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLORS["surface_alt"],
            bordercolor=COLORS["border_strong"],
            font=dict(color=COLORS["text"], size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            borderwidth=0,
            font=dict(color=COLORS["text"], size=12),
        ),
        margin=dict(l=50, r=30, t=40, b=50),
    )
)


def install_template() -> None:
    """Register the dashboard template with Plotly globally."""
    pio.templates["wittgenstein"] = PLOTLY_TEMPLATE
    pio.templates.default = "wittgenstein"


# ---------------------------------------------------------------------------
# Custom CSS - injected on every page
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
    /* Use a more readable system font stack */
    html, body, [class*="css"]  {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Inter", "Helvetica Neue", Arial, sans-serif;
    }}

    /* Page background */
    .stApp {{
        background-color: {COLORS['background']};
    }}

    /* Tighten Streamlit's default padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Force gap between Streamlit columns regardless of testid changes */
    [data-testid="stHorizontalBlock"],
    [class*="stHorizontalBlock"] {{
        gap: 1.25rem !important;
        margin-bottom: 1.25rem !important;
    }}

    /* KPI card - fixed height so all cards in a row are identical
       regardless of how many lines of content they have.
       180px fits the largest case (label + value + delta + sub) with
       a little breathing room. Less-content cards leave empty space
       at the bottom rather than shrinking.
       margin-bottom adds a vertical gap when cards are stacked in the
       same column (e.g. Strategy Tuner's 2x3 baseline grid). */
    .kpi-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1.4rem 1.5rem;
        height: 180px !important;
        margin-bottom: 1rem;
        box-sizing: border-box;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: {COLORS['border_strong']};
        background: {COLORS['surface_alt']};
    }}
    .kpi-label {{
        color: {COLORS['text_muted']};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }}
    .kpi-value {{
        color: {COLORS['text']};
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-sub {{
        color: {COLORS['text_secondary']};
        font-size: 0.95rem;
        margin-top: 0.75rem;
        line-height: 1.5;
        font-weight: 500;
    }}
    .kpi-delta-positive {{
        color: {COLORS['success']};
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }}
    .kpi-delta-negative {{
        color: {COLORS['danger']};
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }}

    /* Section headers */
    .section-header {{
        color: {COLORS['text']};
        font-size: 1.2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid {COLORS['border']};
        letter-spacing: -0.01em;
    }}

    /* Status badges */
    .badge {{
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-success {{ background: rgba(0,200,5,0.12); color: {COLORS['success']}; }}
    .badge-warning {{ background: rgba(255,176,32,0.12); color: {COLORS['warning']}; }}
    .badge-danger  {{ background: rgba(255,59,48,0.12);  color: {COLORS['danger']}; }}
    .badge-info    {{ background: rgba(0,180,216,0.12);  color: {COLORS['secondary']}; }}

    /* Hide hamburger menu, footer, and de-clutter header */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* DataFrame styling */
    .stDataFrame {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    /* Sidebar - slightly distinct from main background */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}

    /* Plotly chart container */
    [data-testid="stPlotlyChart"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        background: {COLORS['surface']};
        padding: 0;
        overflow: visible;
    }}
    /* Plotly modebar - keep inside the frame and styled for dark theme */
    .modebar-container {{
        right: 8px !important;
        top: 8px !important;
    }}
    .modebar-btn svg {{
        fill: {COLORS['text_muted']} !important;
    }}
    .modebar-btn:hover svg {{
        fill: {COLORS['text']} !important;
    }}
    .modebar-btn.active svg {{
        fill: {COLORS['primary']} !important;
    }}

    /* General body text larger and clearer */
    .stMarkdown p {{
        font-size: 0.95rem;
        line-height: 1.65;
        color: {COLORS['text_secondary']};
    }}

    /* Stronger captions */
    [data-testid="stCaptionContainer"] {{
        color: {COLORS['text_muted']};
        font-size: 0.85rem;
    }}

    /* Divider */
    hr {{
        border-color: {COLORS['border']};
    }}
</style>
"""
