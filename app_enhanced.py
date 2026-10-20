"""
AI-Powered Financial Intelligence Platform v3.0 — structural redesign.
Research-backed patterns (Mercury/Ramp/fintech dashboard conventions):
  - one dominant "north star" hero metric instead of a flat KPI row
  - asymmetric bento-weighted grids (variable tile size = importance)
  - custom HTML data tables (sticky header, tabular numerics, status pills)
    instead of Streamlit's default canvas dataframe grid
  - inline SVG sparklines / ring gauges, no chart library overhead
  - grouped, sectioned sidebar navigation instead of one flat list
  - st.tabs() to break up long vertical stacks into real sub-navigation
All data loading / ML / AI logic is unchanged from v2.1.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings, io, base64, os, re, math, zlib, html as _html
warnings.filterwarnings('ignore')

def esc(x):
    """Escape user-controlled text (merchant names, categories from uploaded
    files) before it's embedded in raw HTML — an uploaded CSV shouldn't be
    able to inject markup into the page."""
    return _html.escape(str(x), quote=True)

# ── Load .env before anything reads os.environ (ai_utils_enhanced reads
#    API keys at import time, so this must run first) ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Original AI utilities ─────────────────────────────────────────
try:
    from ai_utils import (
        generate_spending_insights, predict_monthly_spending,
        detect_anomalies_ai, generate_budget_recommendations,
        analyze_subscriptions_ai, generate_financial_health_assessment,
        generate_smart_bill_reminders,
    )
    ORIGINAL_AI_ENABLED = True
except ImportError:
    ORIGINAL_AI_ENABLED = False

# ── Enhanced AI utilities ─────────────────────────────────────────
try:
    from ai_utils_enhanced import (
        chat_with_claude, parse_pdf_statement, parse_image_receipt,
        extract_transaction_from_ocr, generate_realistic_dataset,
        detect_recurring_payments_enhanced, compute_financial_health_score,
        generate_smart_alerts, CHATBOT_ENABLED,
        parse_nayapay_csv, parse_nayapay_pdf_text,
        HF_API_KEY,
    )
    ENHANCED_AI = True
except ImportError:
    ENHANCED_AI = False
    CHATBOT_ENABLED = False
    HF_API_KEY = ""

AI_ENABLED = ORIGINAL_AI_ENABLED

st.set_page_config(page_title="FinanceIQ", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
#  DESIGN TOKENS (python-side, mirrored into CSS below)
# ══════════════════════════════════════════════════════════════════
BG, SURFACE1, SURFACE2 = "#13112a", "#1c1a3a", "#2c2958"
TEXT, TEXT_DIM, TEXT_FAINT = "#f1efff", "#9b97c2", "#6f6a9c"
ACCENT, ACCENT_DIM = "#8b7cf6", "#6d5ee0"
POSITIVE, NEGATIVE, WARNING = "#34d399", "#fb7185", "#fbbf24"
CATEGORY_PALETTE = ['#8b7cf6','#22d3ee','#34d399','#fbbf24','#fb7185','#a78bfa','#f2905a','#60a5fa','#f472b6','#a3e635']

# ══════════════════════════════════════════════════════════════════
#  ICON SET — inline SVG, stroke-based, no emoji anywhere
# ══════════════════════════════════════════════════════════════════
_ICONS = {
    'grid':            '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    'message':         '<path d="M4 5h16v11H8l-4 4V5z"/>',
    'bar-chart':       '<line x1="5" y1="20" x2="5" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="19" y1="20" x2="19" y2="14"/>',
    'shield':          '<path d="M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z"/>',
    'repeat':          '<polyline points="4 7 4 4 9 4"/><path d="M4 7a8 8 0 0113-5"/><polyline points="20 17 20 20 15 20"/><path d="M20 17a8 8 0 01-13 5"/>',
    'pulse':           '<polyline points="3 12 8 12 10 6 14 18 16 12 21 12"/>',
    'bell':            '<path d="M6 8a6 6 0 0112 0c0 5 2 6 2 6H4s2-1 2-6z"/><path d="M10 19a2 2 0 004 0"/>',
    'wallet':          '<path d="M3 7a2 2 0 012-2h13a1 1 0 011 1v3"/><rect x="3" y="7" width="18" height="12" rx="2"/><circle cx="16.5" cy="13" r="1.3" fill="currentColor" stroke="none"/>',
    'calendar':        '<rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="10" x2="21" y2="10"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="16" y1="3" x2="16" y2="7"/>',
    'receipt':         '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2V3z"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="9" y1="12" x2="15" y2="12"/>',
    'camera':          '<path d="M4 8h3l2-2h6l2 2h3v11H4V8z"/><circle cx="12" cy="13.5" r="3.2"/>',
    'download':        '<path d="M12 3v12"/><polyline points="7 11 12 16 17 11"/><path d="M5 19h14"/>',
    'arrow-up-right':  '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="8 7 17 7 17 16"/>',
    'arrow-down-right':'<line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 8 17 17 8 17"/>',
    'check-circle':    '<circle cx="12" cy="12" r="9"/><polyline points="8 12.5 11 15.5 16 9"/>',
    'alert-triangle':  '<path d="M12 4l9 16H3z"/><line x1="12" y1="10" x2="12" y2="14.3"/><circle cx="12" cy="17" r=".9" fill="currentColor" stroke="none"/>',
    'alert-octagon':   '<circle cx="12" cy="12" r="9"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/>',
    'info':            '<circle cx="12" cy="12" r="9"/><line x1="12" y1="10.5" x2="12" y2="16"/><circle cx="12" cy="7.6" r=".9" fill="currentColor" stroke="none"/>',
    'inbox':           '<path d="M4 13h4l2 3h4l2-3h4"/><path d="M5.5 6h13l2.5 7v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6l2.5-7z"/>',
}
def icon(name, size=16, color="currentColor"):
    inner = _ICONS.get(name, _ICONS['info'])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-3px;flex-shrink:0">{inner}</svg>')

def sparkline_svg(values, width=140, height=40, color=ACCENT, fill=True):
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1
    n = len(values)
    pad = 3
    pts = [(pad + i/(n-1)*(width-2*pad), pad + (height-2*pad) - (v-vmin)/rng*(height-2*pad)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = ""
    if fill:
        area_pts = f"{pts[0][0]:.1f},{height} " + line + f" {pts[-1][0]:.1f},{height}"
        area = f'<polyline points="{area_pts}" fill="{color}" opacity="0.10" stroke="none"/>'
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'{area}<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')

def ring_svg(pct, size=88, stroke=8, color=ACCENT, track="rgba(255,255,255,.08)"):
    pct = max(0, min(100, pct))
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    offset = c * (1 - pct/100)
    cx = cy = size/2
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{track}" stroke-width="{stroke}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{c:.2f}" stroke-dashoffset="{offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"/></svg>')

def pill(text, kind="neutral"):
    return f'<span class="pill pill-{kind}">{text}</span>'

def cat_color(cat):
    # zlib.crc32 instead of hash() — Python randomizes str hashing per
    # process (PYTHONHASHSEED), which would make category colors change
    # on every server restart.
    return CATEGORY_PALETTE[zlib.crc32(str(cat).encode()) % len(CATEGORY_PALETTE)]

def cat_tag(cat):
    return f'<span class="cat-tag"><span class="cat-dot" style="background:{cat_color(cat)}"></span>{esc(cat)}</span>'

def amount_cell(amount, ttype):
    sign = '+' if ttype == 'income' else '−'
    cls = ' pos' if ttype == 'income' else ''
    return f'<span class="amt{cls}">{sign} PKR {amount:,.0f}</span>'

def html_table(headers, rows, right_align=None, max_height=None):
    right_align = right_align or set()
    thead = "".join(f'<th class="{"num" if i in right_align else ""}">{h}</th>' for i, h in enumerate(headers))
    body = []
    for row in rows:
        tds = "".join(f'<td class="{"num" if i in right_align else ""}">{cell}</td>' for i, cell in enumerate(row))
        body.append(f'<tr>{tds}</tr>')
    style = f' style="max-height:{max_height}px"' if max_height else ''
    return (f'<div class="table-wrap"{style}><table class="ftable"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')

def empty_state(title, message, icon_name="inbox"):
    st.markdown(f'<div class="empty-state">{icon(icon_name,26,"var(--text-faint)")}'
                f'<div class="empty-title">{title}</div><div class="empty-msg">{message}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CSS — "Ultraviolet": deep indigo + violet / mint, Sora / Inter
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{{
  --bg:{BG}; --surface:#16143a; --surface-1:{SURFACE1}; --surface-2:{SURFACE2};
  --border:rgba(255,255,255,.08); --border-strong:rgba(255,255,255,.16);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --accent:{ACCENT}; --accent-dim:{ACCENT_DIM}; --accent-soft:rgba(139,124,246,.18);
  --positive:{POSITIVE}; --positive-soft:rgba(52,211,153,.12);
  --negative:{NEGATIVE}; --negative-soft:rgba(251,113,133,.14);
  --warning:{WARNING}; --warning-soft:rgba(251,191,36,.13);
  --radius:14px; --radius-sm:9px;
  --font-sans:'Inter',sans-serif; --font-display:'Sora',sans-serif; --font-mono:'JetBrains Mono',monospace;
}}
[data-testid="stDecoration"]{{display:none!important;}}
#MainMenu,footer,.stDeployButton{{visibility:hidden!important;display:none!important;}}
header[data-testid="stHeader"]{{background:transparent!important;border-bottom:none!important;box-shadow:none!important;}}
[data-testid="stSidebarCollapsedControl"]{{display:flex!important;visibility:visible!important;opacity:1!important;background:var(--surface-1)!important;border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;z-index:9999!important;}}
html,body,[class*="css"]{{font-family:var(--font-sans)!important;color:var(--text);}}
.stApp{{background-color:var(--bg)!important;}}
.block-container{{padding:14px 40px 56px!important;max-width:1320px!important;}}
[data-testid="stSidebar"]{{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;padding-top:0!important;width:270px!important;}}
[data-testid="stSidebar"] .block-container{{padding:0!important;}}

@keyframes fadeUp{{from{{opacity:0;transform:translateY(6px);}}to{{opacity:1;transform:translateY(0);}}}}
.section-card,.kpi-card,.hero-card,.side-card{{animation:fadeUp .35s ease-out;}}
@media (prefers-reduced-motion:reduce){{.section-card,.kpi-card,.hero-card,.side-card{{animation:none;}}}}

/* ── Typography rhythm ─────────────────────────────────────────── */
h1,h2,h3,h4{{font-family:var(--font-sans)!important;letter-spacing:-.01em;}}
.page-eyebrow{{font-family:var(--font-mono);font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:6px;}}
.page-title{{font-family:var(--font-display)!important;font-size:21px;font-weight:700;color:var(--text);letter-spacing:-.01em;margin-bottom:2px;}}
.page-subtitle{{font-size:12.5px;color:var(--text-faint);margin-bottom:22px;}}

/* ── Hero card (north-star metric) ───────────────────────────────── */
.hero-card{{background:radial-gradient(120% 130% at 100% 0%,rgba(139,124,246,.16),transparent 60%),var(--surface-1);
  border:1px solid var(--border);border-radius:var(--radius);padding:24px 26px;position:relative;overflow:hidden;height:100%;}}
.hero-label{{font-family:var(--font-mono);font-size:10.5px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--text-faint);}}
.hero-value{{font-family:var(--font-display);font-size:44px;font-weight:800;letter-spacing:-.02em;color:var(--text);margin-top:10px;line-height:1;}}
.hero-delta{{display:inline-flex;align-items:center;gap:4px;font-family:var(--font-mono);font-size:12px;color:var(--text-dim);margin-top:10px;padding:3px 9px;border-radius:20px;background:var(--surface-2);}}
.hero-delta.up{{color:var(--positive);background:var(--positive-soft);}} .hero-delta.down{{color:var(--negative);background:var(--negative-soft);}}
.hero-spark{{margin-top:14px;line-height:0;}}
.hero-sub{{font-size:11.5px;color:var(--text-faint);margin-top:8px;}}

/* ── Side card (alerts + health ring) ────────────────────────────── */
.side-card{{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;height:100%;display:flex;flex-direction:column;gap:16px;}}
.ring-row{{display:flex;align-items:center;gap:16px;}}
.ring-wrap{{position:relative;flex-shrink:0;}}
.ring-num{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:19px;font-weight:700;color:var(--text);}}
.ring-meta-label{{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--text-faint);}}
.ring-meta-grade{{font-size:15px;font-weight:700;color:var(--text);margin-top:3px;}}
.side-note{{font-size:12px;color:var(--text-dim);line-height:1.6;border-top:1px solid var(--border);padding-top:14px;}}

/* ── KPI strip ────────────────────────────────────────────────── */
.kpi-card{{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px;transition:border-color .15s ease;}}
.kpi-card:hover{{border-color:var(--border-strong);}}
.kpi-top{{display:flex;justify-content:space-between;align-items:flex-start;}}
.kpi-label{{font-family:var(--font-mono);font-size:9.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);}}
.kpi-value{{font-family:var(--font-display);font-size:21px;font-weight:600;letter-spacing:-.01em;color:var(--text);font-variant-numeric:tabular-nums;margin-top:8px;}}
.kpi-delta{{display:flex;align-items:center;gap:3px;font-family:var(--font-mono);font-size:10.5px;color:var(--text-faint);margin-top:6px;}}
.kpi-delta.up{{color:var(--positive);}} .kpi-delta.down{{color:var(--negative);}}

/* ── Section cards ─────────────────────────────────────────────── */
.section-card{{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;margin-bottom:16px;}}
.section-card-title{{font-size:13.5px;font-weight:700;color:var(--text);}}
.section-card-subtitle{{font-family:var(--font-mono);font-size:11px;color:var(--text-faint);margin-top:2px;}}

/* ── Custom HTML tables ───────────────────────────────────────────── */
.table-wrap{{border:1px solid var(--border);border-radius:var(--radius);overflow:auto;}}
.ftable{{width:100%;border-collapse:collapse;font-size:12.5px;}}
.ftable thead{{position:sticky;top:0;background:var(--surface-2);z-index:1;}}
.ftable th{{text-align:left;padding:10px 14px;font-family:var(--font-mono);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--text-faint);border-bottom:1px solid var(--border);white-space:nowrap;}}
.ftable td{{padding:10px 14px;border-bottom:1px solid var(--border);color:var(--text-dim);white-space:nowrap;}}
.ftable tbody tr:last-child td{{border-bottom:none;}}
.ftable tbody tr:hover{{background:var(--surface-2);}}
.ftable td.num,.ftable th.num{{text-align:right;}}
.amt{{font-family:var(--font-mono);font-variant-numeric:tabular-nums;color:var(--text);}}
.amt.pos{{color:var(--positive);}}
.cat-tag{{display:inline-flex;align-items:center;gap:6px;color:var(--text-dim);}}
.cat-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}

/* ── Pills / badges ───────────────────────────────────────────────── */
.pill{{display:inline-block;padding:2.5px 9px;border-radius:20px;font-size:10.5px;font-family:var(--font-mono);font-weight:500;white-space:nowrap;}}
.pill-positive{{background:var(--positive-soft);color:var(--positive);}}
.pill-neutral{{background:var(--surface-2);color:var(--text-dim);}}
.pill-danger{{background:var(--negative-soft);color:var(--negative);}}
.pill-warning{{background:var(--warning-soft);color:var(--warning);}}
.pill-accent{{background:var(--accent-soft);color:var(--accent);}}

/* ── Status banners ─────────────────────────────────────────────── */
.status-banner{{display:flex;gap:10px;align-items:flex-start;padding:10px 13px;border-radius:var(--radius-sm);font-size:12.5px;line-height:1.55;margin:8px 0;border-left:3px solid;background:var(--surface-1);}}
.status-banner svg{{margin-top:2px;}}
.status-success{{border-color:var(--positive);color:#aee3c5;}}
.status-warning{{border-color:var(--warning);color:#ecd497;}}
.status-danger{{border-color:var(--negative);color:#f4a7a9;}}
.status-info{{border-color:var(--accent);color:#c4bbff;}}

/* ── Empty states ─────────────────────────────────────────────────── */
.empty-state{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:44px 24px;color:var(--text-faint);text-align:center;}}
.empty-title{{font-weight:700;color:var(--text-dim);font-size:13.5px;}}
.empty-msg{{font-size:12px;max-width:320px;line-height:1.6;}}

/* ── Sidebar brand ────────────────────────────────────────────────── */
.brand{{padding:22px 20px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;}}
.brand-name{{font-family:var(--font-display);font-size:17px;font-weight:700;color:var(--text);letter-spacing:-.01em;}}
.brand-name span{{color:var(--accent);}}
.brand-tag{{font-family:var(--font-mono);font-size:9px;letter-spacing:.16em;text-transform:uppercase;color:var(--text-faint);margin-top:1px;}}

/* ── Sidebar section labels / dividers ───────────────────────────── */
.sidebar-section-label{{font-family:var(--font-mono);font-size:10px!important;font-weight:600!important;letter-spacing:.13em;text-transform:uppercase;color:var(--text-faint)!important;padding:18px 20px 8px;display:block;}}
.sidebar-divider{{height:1px;background:var(--border);margin:6px 0;}}

/* ── Radio-as-nav (data source + page nav), grouped with CSS labels ─── */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child{{display:none!important;}}
[data-testid="stSidebar"] [role="radiogroup"]{{gap:2px;padding:0 10px;}}
[data-testid="stSidebar"] [role="radiogroup"] label{{padding:8px 12px!important;border-radius:var(--radius-sm);margin:0!important;border-left:2px solid transparent;transition:border-color .15s ease,background .15s ease;}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{{background:var(--surface-1);}}
[data-testid="stSidebar"] [role="radiogroup"] label p{{font-family:var(--font-sans)!important;font-size:13.5px!important;font-weight:400;color:var(--text-dim)!important;letter-spacing:0;}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){{background:var(--surface-1);border-left-color:var(--accent);}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{{color:var(--text)!important;font-weight:600!important;}}
[role="radiogroup"][aria-label="nav"]{{margin-top:2px;}}
[role="radiogroup"][aria-label="nav"] label:nth-of-type(2),
[role="radiogroup"][aria-label="nav"] label:nth-of-type(3),
[role="radiogroup"][aria-label="nav"] label:nth-of-type(8),
[role="radiogroup"][aria-label="nav"] label:nth-of-type(11){{margin-top:10px!important;}}

/* ── Sidebar footer stat block ────────────────────────────────────── */
.sb-stats{{font-family:var(--font-mono);font-size:10.5px;color:var(--text-faint);line-height:2.1;padding:10px 20px 2px;}}
.sb-stats strong{{font-weight:600;}}
.sb-dot{{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px;}}
.sb-footer{{padding:16px 20px 20px;margin-top:6px;border-top:1px solid var(--border);}}
.sb-credit{{font-family:var(--font-mono);font-size:10px;color:var(--text-faint);text-align:center;letter-spacing:.02em;}}
.sb-credit b{{color:var(--text-dim);font-weight:600;}}

/* ── Chat ─────────────────────────────────────────────────────── */
.chat-label{{font-family:var(--font-mono);font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin:0 2px 3px;}}
.chat-label.user{{color:var(--text-faint);text-align:right;}}
.chat-label.ai{{color:var(--accent);}}
.chat-user{{background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius) var(--radius) var(--radius-sm) var(--radius);padding:10px 14px;margin:2px 0 12px 15%;font-size:13.5px;color:var(--text);line-height:1.6;}}
.chat-ai{{background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius) var(--radius) var(--radius) var(--radius-sm);padding:10px 14px;margin:2px 15% 12px 0;font-size:13.5px;color:var(--text);line-height:1.6;}}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{{gap:4px;border-bottom:1px solid var(--border);}}
.stTabs [data-baseweb="tab"]{{font-family:var(--font-mono)!important;font-size:12px!important;color:var(--text-faint)!important;padding:8px 4px!important;}}
.stTabs [aria-selected="true"]{{color:var(--text)!important;border-bottom-color:var(--accent)!important;}}
.stTabs [data-baseweb="tab-highlight"]{{background-color:var(--accent)!important;}}

/* ── Streamlit native widget re-skin ─────────────────────────────── */
[data-testid="stMetricValue"]{{font-family:var(--font-mono)!important;font-weight:600!important;font-variant-numeric:tabular-nums;}}
[data-testid="stMetricLabel"]{{font-family:var(--font-mono)!important;font-size:10.5px!important;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint)!important;}}
[data-testid="stMetricDelta"]{{font-family:var(--font-mono)!important;font-size:11.5px!important;}}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{{
  background:var(--surface-2)!important;color:var(--text)!important;border:1px solid var(--border-strong)!important;
  border-radius:var(--radius-sm)!important;font-family:var(--font-sans)!important;font-weight:600;font-size:12.5px!important;
  padding:.5em 1em!important;transition:border-color .15s ease,background .15s ease;box-shadow:none!important;
}}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{{border-color:var(--accent)!important;color:var(--accent)!important;}}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{{background:var(--accent)!important;color:#fff!important;border-color:var(--accent)!important;}}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover{{background:var(--accent-dim)!important;}}
[data-testid="stDataFrame"]{{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;}}
.stTextInput input,.stNumberInput input,.stSelectbox [data-baseweb="select"]>div,.stDateInput input,.stTextArea textarea{{
  font-family:var(--font-mono)!important;background:var(--surface-2)!important;border-color:var(--border-strong)!important;color:var(--text)!important;
}}
[data-baseweb="progress-bar"] > div{{background:var(--surface-2)!important;}}
[data-baseweb="progress-bar"] > div > div{{background:transparent!important;}}
[data-baseweb="progress-bar"] > div > div > div{{background:var(--accent)!important;}}
[data-testid="stFileUploaderDropzone"]{{background:var(--surface-1)!important;border:1px dashed var(--border-strong)!important;border-radius:var(--radius)!important;}}
hr{{border-color:var(--border)!important;}}
/* ── Overview (dashboard) ─────────────────────────────────────── */
[data-testid="column"]:has(.ov-aside){{background:#16143a;border:1px solid var(--border);border-radius:18px;padding:16px 14px!important;}}
.ov-top .t{{font-family:var(--font-display);font-size:22px;font-weight:600;letter-spacing:-.02em;color:var(--text);line-height:1.3;}}
.ov-top p{{color:var(--text-dim);font-size:12px;margin:2px 0 0;}}
[role="radiogroup"][aria-label="Range"]{{display:flex;flex-direction:row;background:var(--surface-1);border-radius:11px;padding:3px;gap:2px;width:max-content;margin-left:auto;}}
[role="radiogroup"][aria-label="Range"] label{{padding:6px 13px!important;border-radius:8px;margin:0!important;cursor:pointer;}}
[role="radiogroup"][aria-label="Range"] label > div:first-child{{display:none!important;}}
[role="radiogroup"][aria-label="Range"] label p{{font-size:12px!important;font-weight:600!important;color:var(--text-dim)!important;}}
[role="radiogroup"][aria-label="Range"] label:has(input:checked){{background:var(--surface-2);}}
[role="radiogroup"][aria-label="Range"] label:has(input:checked) p{{color:var(--text)!important;}}
.ov-bal{{background:linear-gradient(140deg,#2a2463,#1c1a3a 70%);border:1px solid var(--border);border-radius:18px;padding:16px 18px 6px;overflow:hidden;margin-bottom:14px;}}
.ov-bal small{{color:var(--text-dim);font-weight:600;font-size:12px;}}
.ov-bal .row{{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}}
.ov-bal b{{font-family:var(--font-display);font-size:34px;font-weight:600;letter-spacing:-.03em;color:var(--text);}}
.ov-chip{{font-style:normal;font-weight:700;font-size:12px;padding:2px 9px;border-radius:999px;color:var(--positive);background:var(--positive-soft);}}
.ov-chip.down{{color:var(--negative);background:var(--negative-soft);}}
.ov-bal svg{{width:100%;height:150px;display:block;margin-top:6px;}}
.ov-ax{{display:flex;justify-content:space-between;color:var(--text-faint);font-size:10.5px;padding:0 2px 8px;}}
.ov-row2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.ov-c{{background:var(--surface-1);border:1px solid var(--border);border-radius:16px;padding:14px 16px;}}
.ov-h{{font-size:13px;font-weight:700;margin:0 0 10px;display:flex;justify-content:space-between;color:var(--text);}}
.ov-h span{{color:var(--text-dim);font-weight:500;}}
.ov-cat{{display:grid;grid-template-columns:10px 1fr auto;gap:9px;align-items:center;padding:5px 0;font-size:12.5px;color:var(--text);}}
.ov-cat i{{width:10px;height:10px;border-radius:3px;}}
.ov-cat b,.ov-sb b{{font-family:var(--font-display);font-weight:600;font-size:12.5px;}}
.ov-stack{{display:flex;gap:3px;height:10px;margin-top:12px;border-radius:999px;overflow:hidden;}}
.ov-stack i{{display:block;}}
.ov-hint{{margin-top:10px;color:var(--text-dim);font-size:11.5px;}}
.ov-hint b{{color:var(--text);}}
.ov-sb{{display:flex;align-items:center;gap:10px;padding:6px 0;border-top:1px solid var(--border);color:var(--text);font-size:12.5px;}}
.ov-sb:first-of-type{{border-top:0;}}
.ov-sb .lo{{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;font-weight:800;font-size:12px;color:#13112a;flex:none;}}
.ov-sb > span:nth-child(2){{flex:1;min-width:0;}}
.ov-sb small{{display:block;color:var(--text-dim);font-size:10.5px;}}
.ov-tg{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;background:var(--negative-soft);color:var(--negative);margin-left:6px;}}
.ov-tg.info{{background:var(--accent-soft);color:#c4bbff;}}
.ov-empty{{color:var(--text-dim);font-size:12px;padding:10px 0;}}
.ov-ah{{display:flex;align-items:center;gap:10px;margin-bottom:12px;}}
.ov-ah i{{width:32px;height:32px;border-radius:10px;background:linear-gradient(135deg,#8b7cf6,#22d3ee);display:grid;place-items:center;font-style:normal;color:#fff;}}
.ov-ah b{{font-size:14px;display:block;color:var(--text);}}
.ov-ah small{{color:var(--positive);font-size:11px;}}
.ov-ah small.off{{color:var(--warning);}}
.ov-alert{{background:rgba(251,113,133,.1);border:1px solid rgba(251,113,133,.35);border-radius:13px;padding:11px 13px;}}
.ov-alert b{{color:#fecdd3;font-size:12.5px;display:block;margin-bottom:3px;}}
.ov-alert p{{color:#d5d1f2;font-size:11.5px;line-height:1.5;margin:0;}}
.ov-alert.ok{{background:rgba(52,211,153,.08);border-color:rgba(52,211,153,.3);}}
.ov-alert.ok b{{color:#a7f3d0;}}
.ov-msgs{{display:flex;flex-direction:column;gap:8px;margin:12px 0 10px;}}
.ov-msg{{max-width:92%;padding:9px 12px;border-radius:14px;font-size:12px;line-height:1.5;color:var(--text);}}
.ov-msg.me{{align-self:flex-end;background:#3a33a0;border-bottom-right-radius:4px;}}
.ov-msg.ai{{align-self:flex-start;background:var(--surface-1);border:1px solid var(--border);border-bottom-left-radius:4px;}}
.ov-msg.ai b{{color:var(--positive);}}
[data-testid="column"]:has(.ov-aside) .stButton>button{{white-space:nowrap!important;border-radius:999px!important;font-size:11px!important;padding:.3em .7em!important;min-height:0!important;background:transparent!important;border-color:rgba(139,124,246,.45)!important;color:#c4bbff!important;}}
[data-testid="column"]:has(.ov-aside) .stButton>button p{{font-size:11.5px!important;font-weight:600!important;font-family:var(--font-sans)!important;}}
[data-testid="column"]:has(.ov-aside) .stButton>button[kind="primary"]{{background:var(--negative)!important;border-color:var(--negative)!important;color:#2a0a10!important;}}
[data-testid="column"]:has(.ov-aside) [data-testid="stForm"]{{border:1px solid var(--border)!important;border-radius:12px!important;background:var(--surface-1)!important;padding:6px 8px!important;}}
[data-testid="column"]:has(.ov-aside) [data-testid="stForm"] input,
[data-testid="column"]:has(.ov-aside) [data-testid="stForm"] [data-baseweb="input"],
[data-testid="column"]:has(.ov-aside) [data-testid="stForm"] [data-baseweb="base-input"]{{background:transparent!important;border:0!important;font-family:var(--font-sans)!important;font-size:12px!important;}}
[data-testid="column"]:has(.ov-aside) [data-testid="column"]{{min-width:0!important;}}
[data-testid="column"]:has(.ov-aside) [data-testid="stForm"] .stFormSubmitButton>button{{background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important;border-radius:9px!important;}}
@media (max-width:900px){{.ov-row2{{grid-template-columns:1fr;}}}}
</style>""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter',color='#9b97c2',size=11),margin=dict(l=8,r=8,t=8,b=8),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)',zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)',zerolinecolor='rgba(255,255,255,0.08)'))

# ══════════════════════════════════════════════════════════════════
#  LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════
def card_open(title,subtitle=""):
    sub = f'<div class="section-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-card"><div class="section-card-title">{title}</div>{sub}</div>',unsafe_allow_html=True)
def card_close(): pass

_STATUS_ICON = {'success':'check-circle','warning':'alert-triangle','danger':'alert-octagon','info':'info'}
def status(t,m):
    st.markdown(f'<div class="status-banner status-{t}">{icon(_STATUS_ICON.get(t,"info"),15)}<span>{m}</span></div>',unsafe_allow_html=True)

def page_header(t,s="",eyebrow=""):
    if eyebrow: st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{t}</div>',unsafe_allow_html=True)
    if s: st.markdown(f'<div class="page-subtitle">{s}</div>',unsafe_allow_html=True)

def spacer(h=12): st.markdown(f'<div style="height:{h}px"></div>',unsafe_allow_html=True)

def plotly_fig(fig,height=300,**kw):
    fig.update_layout(**{**PLOTLY_LAYOUT,**kw})
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

def kpi_card(label,value,delta="",trend=None):
    d = ""
    if delta:
        arrow = icon('arrow-up-right' if trend=='up' else 'arrow-down-right',10) if trend in ('up','down') else ""
        cls = f" {trend}" if trend in ('up','down') else ""
        d = f'<div class="kpi-delta{cls}">{arrow}<span>{delta}</span></div>'
    return f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-label">{label}</div></div><div class="kpi-value">{value}</div>{d}</div>'

def kpi_row(cards):
    cols = st.columns(len(cards))
    for col,c in zip(cols,cards):
        label,value = c[0],c[1]
        delta = c[2] if len(c)>2 else ""
        trend = c[3] if len(c)>3 else None
        with col: st.markdown(kpi_card(label,value,delta,trend),unsafe_allow_html=True)

def hero_card(label, value, delta_text, trend, spark_values, subtitle=""):
    color = POSITIVE if trend=='up' else (NEGATIVE if trend=='down' else ACCENT)
    spark = sparkline_svg(spark_values, width=260, height=48, color=color) if spark_values else ""
    arrow = icon('arrow-up-right' if trend=='up' else 'arrow-down-right',12) if trend in ('up','down') else ''
    cls = trend if trend in ('up','down') else ''
    sub = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ''
    return (f'<div class="hero-card"><div class="hero-label">{label}</div>'
            f'<div class="hero-value">{value}</div>'
            f'<div class="hero-delta {cls}">{arrow}<span>{delta_text}</span></div>'
            f'<div class="hero-spark">{spark}</div>{sub}</div>')

def side_card(score, grade, note_html=""):
    ring = ring_svg(score, size=82, stroke=7, color=ACCENT)
    note = f'<div class="side-note">{note_html}</div>' if note_html else ''
    return (f'<div class="side-card"><div class="ring-row">'
            f'<div class="ring-wrap">{ring}<div class="ring-num">{score}</div></div>'
            f'<div><div class="ring-meta-label">Health Score</div><div class="ring-meta-grade">Grade {grade}</div></div>'
            f'</div>{note}</div>')

# ══════════════════════════════════════════════════════════════════
#  CORE ML
# ══════════════════════════════════════════════════════════════════
def auto_categorize(m):
    m = str(m).lower()
    kw = {'Groceries':['grocery','imtiaz','carrefour','chase up','metro','alfatah','hyperstar','naheed'],
          'Food & Dining':['restaurant','cafe','burger','pizza','mcdonald','kfc','hardee','foodpanda','cheetay','careem food','biryani'],
          'Utilities':['lesco','gepco','ssgc','sngpl','ptcl','jazz','telenor','zong','ufone','sui','electric','water','utility','internet'],
          'Rent':['rent','landlord','apartment','lease'],
          'Transportation':['careem','indrive','bykea','uber','lyft','pso','shell','caltex','petrol','fuel','parking'],
          'Entertainment':['netflix','spotify','youtube','cinema','movie','steam','disney','nueplex','cinepax','hulu'],
          'Shopping':['daraz','telemart','shophive','amazon','target','walmart','h&m'],
          'Healthcare':['oladoc','shifa','agha khan','hospital','pharmacy','doctor','dental','lab'],
          'Subscriptions':['subscription','adobe','microsoft','icloud','canva','zoom','dropbox','linkedin','gym','membership'],
          'Financial':['nayapay','jazzcash','easypaisa','hbl','meezan','alfalah','bank','transfer']}
    for cat,words in kw.items():
        if any(w in m for w in words): return cat
    return 'Shopping'

@st.cache_data(show_spinner=False)
def detect_fraud(df):
    if len(df)<10:
        d=df.copy(); d['is_fraud']=False; d['fraud_score']=0.0; return d
    X=df[['amount']].copy(); X['dom']=df['date'].dt.day; X['dow']=df['date'].dt.dayofweek
    Xs=StandardScaler().fit_transform(X)
    iso=IsolationForest(contamination=0.05,random_state=42); preds=iso.fit_predict(Xs)
    d=df.copy(); d['is_fraud']=preds==-1; d['fraud_score']=iso.decision_function(Xs); return d

@st.cache_data(show_spinner=False)
def detect_recurring_payments(df):
    if len(df)==0: return pd.DataFrame()
    rec=[]
    for merchant in df['merchant_name'].unique():
        t=df[df['merchant_name']==merchant].sort_values('date')
        if len(t)<2: continue
        a=t['amount'].values; ma=a.mean()
        if ma==0: continue
        cva=a.std()/ma; cvi=0
        if len(t)>=3:
            diffs=np.diff(t['date'].values).astype('timedelta64[D]').astype(int)
            if len(diffs)>0 and diffs.mean()>0: cvi=diffs.std()/diffs.mean()
        if cva<=0.15 and cvi<=0.3:
            rec.append({'merchant_name':merchant,'category':t['category'].iloc[0],'count':len(t),'avg_amount':round(ma,2),'total_annual':round(ma*12,2)})
    return pd.DataFrame(rec).sort_values('total_annual',ascending=False) if rec else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════
def _normalize(df):
    df.columns=df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]','_',regex=True)
    dc=[c for c in df.columns if any(k in c for k in ['date','time','when','day'])]
    dc=dc[0] if dc else df.columns[0]
    df['date']=pd.to_datetime(df[dc],errors='coerce')
    if df['date'].isna().all():
        for c in df.columns:
            a=pd.to_datetime(df[c],errors='coerce')
            if a.notna().sum()>len(df)*.5: df['date']=a; break
    df=df.dropna(subset=['date'])
    if df.empty: return None
    ac=[c for c in df.columns if any(k in c for k in ['amount','sum','total','debit','credit','value','price','cost'])]
    if not ac:
        for c in df.columns:
            if c=='date': continue
            v=pd.to_numeric(df[c],errors='coerce')
            if v.notna().sum()>len(df)*.5: ac=[c]; break
    if not ac: return None
    df['amount']=pd.to_numeric(df[ac[0]],errors='coerce').fillna(0)
    df=df[df['amount']!=0];
    if df.empty: return None
    mc=[c for c in df.columns if any(k in c for k in ['merchant','description','desc','name','payee','vendor','note','memo','narration','title'])]
    df['merchant_name']=df[mc[0]].astype(str).str.strip() if mc else 'Unknown'
    cc=[c for c in df.columns if 'categ' in c or c in ['type','tag']]
    df['category']=df[cc[0]].astype(str).str.strip() if (cc and cc[0]!='transaction_type') else df['merchant_name'].apply(auto_categorize)
    df['description']=df['merchant_name']
    tc=[c for c in df.columns if 'transaction_type' in c or c in ['type','txn_type','kind']]
    if tc:
        im=df[tc[0]].astype(str).str.strip().str.lower().isin(['income','credit','deposit','salary','revenue'])
    else:
        im=df['amount']<0
    df['amount']=df['amount'].abs()
    df['transaction_type']=np.where(im,'income','expense')
    df=df[['date','merchant_name','category','amount','description','transaction_type']].sort_values('date').reset_index(drop=True)
    df['transaction_id']=[f'TXN{i:06d}' for i in range(len(df))]
    return df

@st.cache_data(show_spinner=False)
def load_sample_data():
    try:
        df=pd.read_csv('sample_transactions.csv')
        df['date']=pd.to_datetime(df['date']); df['amount']=pd.to_numeric(df['amount'],errors='coerce')
        df=df.dropna(subset=['amount'])
        im=(df['transaction_type'].astype(str).str.strip().str.lower()=='income' if 'transaction_type' in df.columns else df['amount']<0)
        df['amount']=df['amount'].abs(); df['transaction_type']=np.where(im,'income','expense')
        if 'merchant_name' not in df.columns: df['merchant_name']=df.get('description','Unknown')
        if 'category' not in df.columns: df['category']=df['merchant_name'].apply(auto_categorize)
        if 'transaction_id' not in df.columns: df['transaction_id']=[f'TXN{i:06d}' for i in range(len(df))]
        return df[df['amount']>0].sort_values('date').reset_index(drop=True)
    except Exception: return None

@st.cache_data(show_spinner=False)
def load_user_data(uploaded_file):
    """Smart loader: NayaPay CSV, standard CSV, Excel."""
    try:
        n = uploaded_file.name.lower()
        raw_bytes = uploaded_file.read()
        if n.endswith('.csv'):
            raw_text = raw_bytes.decode('utf-8', errors='replace')
            if any(k in raw_text for k in ['TIMESTAMP','NayaPay','Peer to Peer','IBFT']):
                if ENHANCED_AI:
                    try:
                        df = parse_nayapay_csv(raw_bytes)
                        if df is not None and len(df) > 0:
                            return df
                    except Exception: pass
                import io as _io
                lines = raw_text.split('\n')
                start = 0
                for i, l in enumerate(lines):
                    if ('TIMESTAMP' in l or 'TIME' in l) and 'AMOUNT' in l:
                        start = i; break
                df = pd.read_csv(_io.StringIO('\n'.join(lines[start:])))
                if 'AMOUNT' in df.columns:
                    df['AMOUNT'] = (df['AMOUNT'].astype(str)
                                   .str.replace(r'Rs\.?\s*','',regex=True)
                                   .str.replace(',','').str.replace('+','').str.strip())
                    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
                if 'DESCRIPTION' in df.columns:
                    df['DESCRIPTION'] = df['DESCRIPTION'].astype(str).str.split('|').str[0].str.split('\n').str[0].str.strip()
                r = _normalize(df)
                if r is not None: return r
            import io as _io
            df = pd.read_csv(_io.BytesIO(raw_bytes))
            if df.empty: st.error("File is empty"); return None
            r = _normalize(df)
            if r is None: st.error("Could not parse date/amount columns."); return None
            return r
        elif n.endswith(('.xlsx', '.xls')):
            import io as _io
            df = pd.read_excel(_io.BytesIO(raw_bytes))
            if df.empty: st.error("File is empty"); return None
            r = _normalize(df)
            if r is None: st.error("Could not parse date/amount columns."); return None
            return r
        else:
            st.error("Please upload CSV, Excel (.xlsx/.xls), or PDF"); return None
    except Exception as e:
        st.error(f"Error reading file: {e}"); return None


@st.cache_data(show_spinner=False)
def load_pdf_data(file_bytes):
    """Parse NayaPay PDF and generic bank PDF into DataFrame."""
    try:
        import pdfplumber, io as _io
        full_text = ''; table_records = []
        with pdfplumber.open(_io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                pt = page.extract_text() or ''
                full_text += pt + '\n'
                for tbl in page.extract_tables():
                    if not tbl or len(tbl)<2: continue
                    hdrs=[str(h).lower().strip().replace(' ','_') if h else f'c{i}' for i,h in enumerate(tbl[0])]
                    for row in tbl[1:]:
                        if row and any(c for c in row if c):
                            table_records.append(dict(zip(hdrs,[str(c).strip() if c else '' for c in row])))

        if any(k in full_text for k in ['NayaPay','Peer to Peer','IBFT In','Raast','hunainriasat']):
            TYPES = ['Peer to Peer','IBFT In','IBFT Out','Raast In','Raast Out',
                     'Online','Cash In','Cash Out','Request In','Request Out','Bill Payment']
            INCOME_T = {'IBFT In','Raast In','Cash In','Request In'}
            line_pat = re.compile(
                r'^(\d{2} \w{3} \d{4})\s+(.+?)\s+([+-]?Rs\.?\s*[\d,]+\.?\d*)\s+(Rs\.?\s*[\d,]+\.?\d*)\s*$'
            )
            records = []
            for line in full_text.split('\n'):
                m = line_pat.match(line.strip())
                if not m: continue
                date_str, middle, amt_str, _ = m.groups()
                txn_type='Unknown'; desc=middle.strip()
                for t in TYPES:
                    if middle.startswith(t): txn_type=t; desc=middle[len(t):].strip(); break
                sign = -1 if '-' in amt_str else 1
                clean = re.sub(r'[Rs\.\s,]','', amt_str).replace('+','').replace('-','').strip()
                try: amount = float(clean) * sign
                except: continue
                is_inc = (sign > 0 or txn_type in INCOME_T)
                merchant = desc.split('|')[0].split('\n')[0].strip()[:60]
                records.append({
                    'date': date_str, 'merchant_name': merchant,
                    'category': _auto_cat(merchant),
                    'amount': abs(amount), 'description': desc,
                    'transaction_type': 'income' if is_inc else 'expense',
                })
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
                return df.sort_values('date').reset_index(drop=True), None

        if table_records:
            df = pd.DataFrame(table_records); r = _normalize(df)
            if r is not None and len(r)>0: return r, None

        return None, "Could not extract transactions. Try exporting as CSV from your bank app instead."
    except ImportError: return None, "pdfplumber not installed: pip install pdfplumber"
    except Exception as e: return None, f"PDF error: {e}"


def _auto_cat(desc):
    m = str(desc).lower()
    cats = {
        'Food & Dining':['foodpanda','kfc','pizza','burger','cafe','restaurant','biryani','cheetay'],
        'Utilities':['lesco','gepco','jazz','telenor','zong','ufone','ptcl','electric','sui gas'],
        'Transportation':['careem','indrive','pso','shell','petrol','bykea'],
        'Entertainment':['netflix','spotify','cinema','steam','skin.club','gaming','skin'],
        'Shopping':['daraz','telemart','amazon'],
        'Financial':['nayapay','jazzcash','easypaisa','hbl','meezan','ibft','raast','transfer'],
    }
    for cat,words in cats.items():
        if any(w in m for w in words): return cat
    return 'Other'


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k,v in [('chat_history',[]),('pending_prompt',None),('generated_df',None),('data_label','sample'),('ocr_transactions',[]),('dismissed_fraud',[])]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
LOGOMARK = ('<svg width="26" height="26" viewBox="0 0 26 26" fill="none">'
            '<rect x="1" y="1" width="24" height="24" rx="7" stroke="var(--accent)" stroke-width="1.6"/>'
            '<path d="M8 17V9h7M8 13h5" stroke="var(--accent)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
st.sidebar.markdown(f"""
<div class="brand">{LOGOMARK}
  <div><div class="brand-name">Finance<span>IQ</span></div><div class="brand-tag">Financial Intelligence</div></div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown('<span class="sidebar-section-label">Data Source</span>',unsafe_allow_html=True)

DATA_OPTIONS=["Upload Bank Statement","Sample Data","Generate Demo Data"]
data_choice=st.sidebar.radio("dsr",DATA_OPTIONS,label_visibility="collapsed",key="data_source")

df=None

if data_choice==DATA_OPTIONS[0]:
    uf=st.sidebar.file_uploader("uf",type=["csv","xlsx","xls","pdf"],label_visibility="collapsed")
    if uf:
        if uf.name.lower().endswith('.pdf'):
            with st.spinner("Parsing PDF…"):
                df,err=load_pdf_data(uf.read())
            if df is not None:
                st.sidebar.success(f"PDF parsed — {len(df):,} transactions")
                st.session_state.data_label='live'
            else:
                st.sidebar.error(f"{err}")
                st.sidebar.caption("Tip: export as CSV from your bank app → Transaction History → Download")
        else:
            with st.spinner("Loading…"): df=load_user_data(uf)
            if df is not None:
                st.sidebar.success(f"Loaded {len(df):,} real transactions")
                st.session_state.data_label='live'

elif data_choice==DATA_OPTIONS[1]:
    df=load_sample_data()
    if df is not None:
        st.sidebar.success(f"Sample data — {len(df):,} transactions")
        st.session_state.data_label='sample'
    else:
        st.sidebar.warning("sample_transactions.csv not found in project folder.")

elif data_choice==DATA_OPTIONS[2]:
    ca,cb=st.sidebar.columns(2)
    with ca: gm=st.number_input("Months",1,12,3,key="gm")
    with cb: gi=st.number_input("Salary",20000,500000,80000,step=5000,key="gi")
    if st.sidebar.button("Generate",use_container_width=True,type="primary"):
        if ENHANCED_AI:
            with st.spinner("Generating…"):
                df=generate_realistic_dataset(months=int(gm),monthly_income=float(gi))
            st.session_state.generated_df=df; st.session_state.data_label='ai'
            st.sidebar.success(f"Generated {len(df):,} transactions")
        else:
            st.sidebar.error("ai_utils_enhanced.py not found")
    if st.session_state.generated_df is not None: df=st.session_state.generated_df

if df is None:
    df=load_sample_data()
    if df is None:
        rng=np.random.default_rng(0); n=80
        df=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=n,freq='4D'),
            'merchant_name':rng.choice(['Netflix','Foodpanda','LESCO','Careem','Daraz','Monthly Salary'],n),
            'amount':rng.uniform(200,8000,n).round(2),
            'transaction_type':np.where(np.arange(n)%10==0,'income','expense'),
            'description':['Transaction']*n})
        df['category']=df['merchant_name'].apply(auto_categorize)
        df['transaction_id']=[f'TXN{i:06d}' for i in range(n)]

if st.session_state.ocr_transactions:
    ocr_df = pd.DataFrame(st.session_state.ocr_transactions)
    ocr_df['date'] = pd.to_datetime(ocr_df['date'], errors='coerce')
    ocr_df = ocr_df.dropna(subset=['date'])
    if len(ocr_df) > 0:
        df = pd.concat([df, ocr_df], ignore_index=True).sort_values('date').reset_index(drop=True)

df_fraud=detect_fraud(df)
recurring_df=(detect_recurring_payments_enhanced(df) if ENHANCED_AI else detect_recurring_payments(df))

# Nav — grouped via CSS label injection (Overview / Assistant / Analyze / Plan / Tools)
NAV_ITEMS=["Dashboard","AI Chat Assistant","Analytics","Fraud Detection",
    "Recurring Payments","Financial Health","Smart Alerts","Budget Planning",
    "Monthly Report","Bill Reminders","OCR Receipt Upload","Reports & Export"]

st.sidebar.markdown('<div class="sidebar-divider"></div>',unsafe_allow_html=True)
st.sidebar.markdown('<span class="sidebar-section-label">Navigation</span>',unsafe_allow_html=True)
page=st.sidebar.radio("nav",NAV_ITEMS,label_visibility="collapsed",key="navigation")
st.sidebar.markdown('<div class="sidebar-divider"></div>',unsafe_allow_html=True)

badge={'live':('Live Bank Data',POSITIVE),'sample':('Sample Data',ACCENT),'ai':('AI-Generated',WARNING)}
bl,bc=badge.get(st.session_state.data_label,badge['sample'])
fc=int(df_fraud['is_fraud'].sum())
ai_status = 'Claude AI' if os.environ.get('ANTHROPIC_API_KEY') else ('HuggingFace AI' if os.environ.get('HUGGINGFACE_API_KEY') else 'Offline mode')
ai_color = POSITIVE if CHATBOT_ENABLED else WARNING
st.sidebar.markdown(f"""<div class="sb-stats">
    <div><span class="sb-dot" style="background:{bc}"></span>Source&nbsp; <strong style="color:{bc}">{bl}</strong></div>
    <div>Transactions&nbsp; <strong style="color:var(--text-dim)">{len(df):,}</strong></div>
    <div>Fraud alerts&nbsp; <strong style="color:{NEGATIVE if fc>0 else 'var(--text-dim)'}">{fc}</strong></div>
    <div>Subscriptions&nbsp; <strong style="color:var(--text-dim)">{len(recurring_df)}</strong></div>
    <div>AI chat&nbsp; <strong style="color:{ai_color}">{ai_status}</strong></div>
</div>
<div class="sb-footer">
  <div class="sb-credit">Built by <b>Zain</b>, <b>Hunain</b> &amp; <b>Shaheer</b></div>
</div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  OVERVIEW (dashboard hero) — balance trend, spending, subscriptions
#  and the finance-assistant panel, all computed from the loaded data
# ══════════════════════════════════════════════════════════════════
RANGE_DAYS = {"Week": 7, "Month": 30, "Year": 365}
CYCLE_DAYS = {"Weekly": 7, "Bi-weekly": 14, "Monthly": 30, "Quarterly": 91, "Annual": 365}

def goto_page(page_name, prompt=None):
    """Button callback: switch page (and optionally queue a chat prompt).
    Runs before the rerun, so it can safely change the nav radio's state."""
    st.session_state.navigation = page_name
    if prompt:
        st.session_state.pending_prompt = prompt

def dismiss_fraud(txn_id):
    st.session_state.dismissed_fraud = list(st.session_state.get('dismissed_fraud', [])) + [txn_id]

def ask_from_overview():
    q = str(st.session_state.get('ov_question', '')).strip()
    if q:
        goto_page("AI Chat Assistant", q)

def _smooth_path(pts):
    """Catmull-Rom spline through the points, as an SVG cubic path."""
    if len(pts) < 2:
        return ""
    d = f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
    return d

def balance_chart_svg(series, rng):
    """Gradient area chart of the running balance with the period's peak marked."""
    W, H, top, bottom = 560, 150, 30, 8
    vals = [float(v) for v in series.values]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1
    n = len(vals)
    pts = [(i / (n - 1) * W, top + (H - top - bottom) * (1 - (v - vmin) / span)) for i, v in enumerate(vals)]
    line = _smooth_path(pts)
    area = f"{line} L{W} {H} L0 {H}Z"
    k = int(np.argmax(vals)); px, py = pts[k]
    label = f"{'−' if vals[k] < 0 else ''}PKR {abs(vals[k]):,.0f} · {series.index[k]:%d %b}"
    lw = 9 + len(label) * 6.1
    lx = min(max(px - lw / 2, 2), W - lw - 2)
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" aria-label="Balance over the last {rng.lower()}">'
            '<defs><linearGradient id="ovg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8b7cf6" stop-opacity=".55"/>'
            '<stop offset="1" stop-color="#8b7cf6" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{area}" fill="url(#ovg)"/>'
            f'<path d="{line}" fill="none" stroke="#a99cff" stroke-width="2.4" vector-effect="non-scaling-stroke"/>'
            f'<line x1="{px:.1f}" y1="{top - 2}" x2="{px:.1f}" y2="{H}" stroke="rgba(255,255,255,.18)" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#13112a" stroke="#c4bbff" stroke-width="2.5" vector-effect="non-scaling-stroke"/>'
            f'<rect x="{lx:.1f}" y="2" width="{lw:.1f}" height="22" rx="7" fill="#2c2958"/>'
            f'<text x="{lx + lw / 2:.1f}" y="17" fill="#f1efff" font-family="Sora,sans-serif" font-size="11" text-anchor="middle">{label}</text></svg>')

def axis_labels(index, rng):
    picks = np.linspace(0, len(index) - 1, 6).round().astype(int)
    fmt = {"Week": "%a", "Month": "%d %b", "Year": "%b"}[rng]
    return "".join(f"<span>{index[i]:{fmt}}</span>" for i in picks)

def monthly_cost(row):
    cycle = row.get('billing_cycle', 'Monthly') if hasattr(row, 'get') else 'Monthly'
    return float(row['avg_amount']) * 30 / CYCLE_DAYS.get(cycle, 30)

def render_overview(df, df_fraud, recurring_df):
    end = df['date'].max().normalize()
    head_l, head_r = st.columns([3, 1.4])
    with head_l:
        st.markdown(f'<div class="ov-top"><div class="t">Overview</div><p>All accounts · {len(df):,} transactions · '
                    f'updated {end:%d %b %Y}</p></div>', unsafe_allow_html=True)
    with head_r:
        rng = st.radio("Range", list(RANGE_DAYS), index=1, horizontal=True,
                       label_visibility="collapsed", key="ov_range")
    days = RANGE_DAYS[rng]
    start = end - pd.Timedelta(days=days - 1)
    period_word = {"Week": "this week", "Month": "this month", "Year": "this year"}[rng]

    # Running balance over the whole history, so the period's figures are true balances.
    signed = np.where(df['transaction_type'] == 'income', df['amount'], -df['amount'])
    daily = pd.Series(signed, index=df['date'].dt.normalize()).groupby(level=0).sum()
    full_idx = pd.date_range(min(daily.index.min(), start - pd.Timedelta(days=1)), end, freq='D')
    running = daily.reindex(full_idx, fill_value=0).cumsum()
    window = running.loc[start:end]
    if rng == "Year":
        window = window.resample('W').last()
    balance = float(running.iloc[-1])
    opening = float(running.loc[:start - pd.Timedelta(days=1)].iloc[-1]) if (running.index < start).any() else 0.0
    change = balance - opening
    # A percentage is only meaningful against a balance at least as large as the move.
    if abs(opening) >= max(abs(change), 1):
        delta_txt = f"{'▲' if change >= 0 else '▼'} {abs(change / abs(opening) * 100):.1f}% {period_word}"
    else:
        delta_txt = f"{'▲' if change >= 0 else '▼'} PKR {abs(change):,.0f} {period_word}"
    chip = "ov-chip" + ("" if change >= 0 else " down")
    chart = balance_chart_svg(window, rng) if len(window) >= 2 else ""
    axis = axis_labels(window.index, rng) if len(window) >= 2 else ""
    st.markdown(f'<div class="ov-bal"><small>Net balance</small><div class="row"><b>{"−" if balance < 0 else ""}PKR {abs(balance):,.0f}</b>'
                f'<em class="{chip}">{delta_txt}</em></div>{chart}<div class="ov-ax">{axis}</div></div>',
                unsafe_allow_html=True)

    # Spending by category vs. a budget of 92% of average income (the
    # Budget Planning page's recommended split, minus its 8% savings line).
    ex = df[(df['transaction_type'] == 'expense') & (df['date'] >= start)]
    spent = float(ex['amount'].sum())
    months_span = max(1.0, (df['date'].max() - df['date'].min()).days / 30.4)
    income_total = float(df[df['transaction_type'] == 'income']['amount'].sum())
    budget = income_total / months_span * 0.92 * days / 30.4
    by_cat = ex.groupby('category')['amount'].sum().sort_values(ascending=False)
    top = by_cat.head(4)
    # Category colours stay consistent app-wide, but two hashed colours can collide;
    # inside one small legend each category must be distinguishable.
    colors, used_c = {}, set()
    for c in top.index:
        col = cat_color(c)
        if col in used_c:
            col = next((x for x in CATEGORY_PALETTE if x not in used_c), col)
        colors[c] = col; used_c.add(col)
    cat_rows = "".join(f'<div class="ov-cat"><i style="background:{colors[c]}"></i>{esc(c)}<b>PKR {v:,.0f}</b></div>'
                       for c, v in top.items()) or '<div class="ov-empty">No spending in this period.</div>'
    scale = max(budget, spent, 1)
    segs = "".join(f'<i style="flex:{v / scale:.4f};background:{colors[c]}"></i>' for c, v in top.items())
    other = spent - float(top.sum())
    if other > 0:
        segs += f'<i style="flex:{other / scale:.4f};background:#6f6a9c"></i>'
    if budget > spent:
        segs += f'<i style="flex:{(budget - spent) / scale:.4f};background:#3a3670"></i>'
    if budget > 0:
        used = spent / budget * 100
        left = budget - spent
        hint = (f'Budget used: <b>{used:.0f}%</b> · PKR {left:,.0f} left {period_word}' if left >= 0
                else f'Budget used: <b>{used:.0f}%</b> · PKR {-left:,.0f} over {period_word}')
    else:
        hint = 'Add income transactions to track a budget.'

    # Subscriptions: the three biggest recurring charges, with the next
    # renewal and a flag on ones that have stopped charging.
    subs_html = '<div class="ov-empty">No recurring charges detected yet.</div>'
    n_subs = 0
    if recurring_df is not None and len(recurring_df) > 0:
        rec = recurring_df.copy()
        rec['monthly'] = rec.apply(monthly_cost, axis=1)
        rec = rec.sort_values('monthly', ascending=False)
        n_subs = len(rec)
        rows = []
        for _, r in rec.head(3).iterrows():
            cyc = CYCLE_DAYS.get(r.get('billing_cycle', 'Monthly'), 30)
            last = pd.to_datetime(r.get('last_charge', end))
            idle = (end - last).days
            nxt = last + pd.Timedelta(days=cyc)
            tag = f'<span class="ov-tg">no charge {idle}d</span>' if idle > cyc * 1.5 else ''
            name = str(r['merchant_name'])
            rows.append(f'<div class="ov-sb"><span class="lo" style="background:{cat_color(r["category"])}">{esc(name[:1].upper())}</span>'
                        f'<span>{esc(name)}{tag}<small>{"Lapsed" if tag else "Renews " + format(nxt, "%d %b")}</small></span>'
                        f'<b>PKR {r["avg_amount"]:,.0f}</b></div>')
        subs_html = "".join(rows)
    st.markdown(f'<div class="ov-row2"><div class="ov-c"><div class="ov-h">Spending by category <span>PKR {spent:,.0f}</span></div>'
                f'{cat_rows}<div class="ov-stack">{segs}</div><p class="ov-hint">{hint}</p></div>'
                f'<div class="ov-c"><div class="ov-h">Subscriptions <span>{n_subs} active</span></div>{subs_html}</div></div>',
                unsafe_allow_html=True)
    return rng, start, end, by_cat, recurring_df

def render_assistant(df_fraud, recurring_df, by_cat, period_word):
    online = CHATBOT_ENABLED
    st.markdown('<div class="ov-aside"></div><div class="ov-ah"><i>✦</i><span><b>Finance assistant</b>'
                f'<small class="{"" if online else "off"}">● {"online" if online else "offline mode"}</small></span></div>',
                unsafe_allow_html=True)

    dismissed = set(st.session_state.get('dismissed_fraud', []))
    # Only outgoing money can be a fraudulent charge (an unusually large salary is not),
    # and a known recurring bill such as rent is expected, however large.
    known = set(recurring_df['merchant_name']) if recurring_df is not None and len(recurring_df) else set()
    flagged = df_fraud[df_fraud['is_fraud'] & (df_fraud['transaction_type'] == 'expense')
                       & ~df_fraud['merchant_name'].isin(known)
                       & ~df_fraud['transaction_id'].isin(dismissed)].sort_values('fraud_score')
    if len(flagged) > 0:
        f = flagged.iloc[0]
        spend = df_fraud.loc[df_fraud['transaction_type'] == 'expense', 'amount']
        larger_than = int((spend < f['amount']).mean() * 100)
        st.markdown(f'<div class="ov-alert"><b>⚠ Unusual charge flagged</b><p>PKR {f["amount"]:,.0f} at '
                    f'{esc(f["merchant_name"])} on {f["date"]:%d %b} — larger than {larger_than}% of your charges '
                    f'and outside your usual pattern.</p></div>',
                    unsafe_allow_html=True)
        a1, a2, _ = st.columns([1, 1, .4])
        with a1:
            st.button("Review", key="ov_review", type="primary", on_click=goto_page, args=("Fraud Detection",))
        with a2:
            st.button("It was me", key="ov_dismiss", on_click=dismiss_fraud, args=(f['transaction_id'],))
    else:
        st.markdown('<div class="ov-alert ok"><b>✓ No unusual charges</b><p>Every transaction in this data set looks normal.</p></div>',
                    unsafe_allow_html=True)

    history = st.session_state.get('chat_history', [])
    if len(history) >= 2:
        q = history[-2]['content']; a = history[-1]['content']
        a = ' '.join(str(a).replace('- ', '').split())
        a = esc(a if len(a) < 240 else a[:237].rsplit(' ', 1)[0] + '…')
        a = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', a)
        q = esc(q)
    else:
        q = f"Where can I save money {period_word}?"
        if recurring_df is not None and len(recurring_df) > 0:
            rec = recurring_df.copy(); rec['monthly'] = rec.apply(monthly_cost, axis=1)
            rec = rec.sort_values('monthly', ascending=False)
            pair = rec.head(2); m2 = float(pair['monthly'].sum())
            names = " and ".join(esc(n) for n in pair['merchant_name'])
            a = (f"You're paying <b>{len(rec)} recurring charges</b> — PKR {rec['monthly'].sum():,.0f}/month. "
                 f"The two largest, {names}, cost <b>PKR {m2:,.0f}/month</b> — that's PKR {m2 * 12:,.0f} a year.")
        elif len(by_cat) > 0:
            c, v = by_cat.index[0], float(by_cat.iloc[0])
            a = (f"Your biggest spend {period_word} is <b>{esc(c)}</b> at PKR {v:,.0f}. "
                 f"Trimming it by 10% saves <b>PKR {v * .1:,.0f}</b>.")
        else:
            a = "Load a statement and I'll show where your money goes."
    st.markdown(f'<div class="ov-msgs"><div class="ov-msg me">{q}</div><div class="ov-msg ai">{a}</div></div>',
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns([1.35, 1.2, .75])
    with s1: st.button("Subscriptions", key="ov_s1", on_click=goto_page, args=("Recurring Payments",), use_container_width=True)
    with s2: st.button("Show budget", key="ov_s2", on_click=goto_page, args=("Budget Planning",), use_container_width=True)
    with s3: st.button("Bills", key="ov_s3", on_click=goto_page, args=("Bill Reminders",), use_container_width=True)

    with st.form("ov_ask", clear_on_submit=True):
        i1, i2 = st.columns([5, 1])
        with i1:
            st.text_input("Ask", placeholder="Ask about your money…", label_visibility="collapsed", key="ov_question")
        with i2:
            st.form_submit_button("↑", on_click=ask_from_overview, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════

# ── DASHBOARD ────────────────────────────────────────────────────
if page=="Dashboard":
    ov_main, ov_aside = st.columns([2.15, 1], gap="medium")
    with ov_main:
        ov_rng, ov_start, ov_end, ov_by_cat, _ = render_overview(df, df_fraud, recurring_df)
    with ov_aside:
        render_assistant(df_fraud, recurring_df, ov_by_cat,
                         {"Week": "this week", "Month": "this month", "Year": "this year"}[ov_rng])

    ti=df[df['transaction_type']=='income']['amount'].sum()
    ts=df[df['transaction_type']=='expense']['amount'].sum()
    health=(compute_financial_health_score(df,ti,ts,recurring_df) if ENHANCED_AI else generate_financial_health_assessment(df,ti,ts))

    spacer(18)
    card_open("Recent Transactions","Last 20 entries, most recent first")
    recent = df.tail(20).iloc[::-1]
    rows = [[r['date'].strftime('%b %d'), esc(r['merchant_name']), cat_tag(r['category']),
             pill('Income','positive') if r['transaction_type']=='income' else pill('Expense','neutral'),
             amount_cell(r['amount'], r['transaction_type'])] for _,r in recent.iterrows()]
    st.markdown(html_table(['Date','Merchant','Category','Type','Amount'], rows, right_align={4}, max_height=420), unsafe_allow_html=True)
    card_close()

    spacer(8)
    card_open("AI-Powered Insights")
    if AI_ENABLED:
        t1,t2,t3 = st.tabs(["Health Breakdown","Spending Analysis","Forecast"])
        with t1:
            if health.get('sub_scores'):
                for k,(lbl,mx) in [('savings',('Savings Ratio',30)),('consistency',('Consistency',20)),
                    ('subscription',('Subscription Load',20)),('risk',('Risk Profile',15)),('income_stability',('Income Stability',15))]:
                    v=health['sub_scores'].get(k,0)
                    st.markdown(f"**{lbl}** — {v:.1f}/{mx}")
                    st.progress(min(1.0,v/mx))
            if '_recommendations' in health: status('info',health['_recommendations'])
        with t2:
            with st.spinner("…"): st.write(generate_spending_insights(df))
        with t3:
            with st.spinner("…"):
                fc2=predict_monthly_spending(df,months_ahead=3)
                if 'error' not in fc2:
                    for k,v in fc2.get('forecasts',{}).items():
                        if not str(k).startswith('_') and isinstance(v,(int,float,np.integer,np.floating)):
                            st.write(f"– {k}: PKR {v:,.0f}")
                    if '_interpretation' in fc2: status('info',fc2['_interpretation'])
                else: st.info("Need 3+ months data for forecast")
    else:
        status('info','Set <code>ANTHROPIC_API_KEY</code> for AI insights.')
    card_close()

# ── AI CHAT ──────────────────────────────────────────────────────
elif page=="AI Chat Assistant":
    page_header("AI Financial Assistant","Ask anything about your finances in plain English",eyebrow="Assistant")

    CLAUDE_KEY2 = os.environ.get("ANTHROPIC_API_KEY","")
    HF_KEY2     = os.environ.get("HUGGINGFACE_API_KEY","")
    if CLAUDE_KEY2:
        status('success','Claude AI connected — your transaction data is loaded as full context.')
    elif HF_KEY2:
        status('success','HuggingFace AI connected (free tier) — Mistral-7B financial assistant active.')
    elif ENHANCED_AI:
        status('warning',
            'Offline mode — smart rule-based answers.<br>'
            '<strong>To enable free AI:</strong> get a free key at '
            '<a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a> '
            'and add <code>HUGGINGFACE_API_KEY=hf_...</code> to your .env file<br>'
            '<strong>Or for Claude:</strong> add <code>ANTHROPIC_API_KEY=sk-ant-...</code>')
    else:
        status('info','Basic mode. Ensure ai_utils_enhanced.py is present.')

    spacer(8)

    if st.session_state.pending_prompt:
        msg=st.session_state.pending_prompt
        st.session_state.pending_prompt=None
        with st.spinner("Thinking…"):
            reply=(chat_with_claude(msg,st.session_state.chat_history,df) if ENHANCED_AI
                   else f"Total expenses: PKR {df[df['transaction_type']=='expense']['amount'].sum():,.0f}. Set ANTHROPIC_API_KEY for full AI answers.")
        st.session_state.chat_history.append({'role':'user','content':msg})
        st.session_state.chat_history.append({'role':'assistant','content':reply})

    card_open("Quick Questions","Click any question for an instant answer")
    QP=["How much did I spend last month?","Which subscriptions are unnecessary?",
        "What category consumes most money?","How can I save more money?",
        "Which transactions look suspicious?","Predict my next month spending"]
    cols=st.columns(3)
    for i,p in enumerate(QP):
        with cols[i%3]:
            if st.button(p,use_container_width=True,key=f"qp_{i}"):
                st.session_state.pending_prompt=p
                st.rerun()
    card_close()

    card_open("Conversation")
    if not st.session_state.chat_history:
        st.markdown("""<div class="chat-label ai">Assistant</div>
<div class="chat-ai">Hi — I'm your AI financial assistant and I have access to your transaction data.<br><br>
I can help you:<br>
– Understand where your money is going<br>
– Find savings and cut unnecessary subscriptions<br>
– Detect unusual or suspicious spending<br>
– Forecast next month's expenses<br><br>
Click a quick question above or type your own below.</div>""",unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn['role']=='user':
            st.markdown(f'<div class="chat-label user">You</div><div class="chat-user">{turn["content"]}</div>',unsafe_allow_html=True)
        else:
            content=turn['content'].replace('\n','<br>').replace('•','&#8211;')
            st.markdown(f'<div class="chat-label ai">Assistant</div><div class="chat-ai">{content}</div>',unsafe_allow_html=True)
    card_close()

    with st.form("chat_form",clear_on_submit=True):
        ui=st.text_input("Question…",placeholder="e.g. How much did I spend on food this month?",label_visibility="collapsed")
        xc1,xc2=st.columns([5,1])
        with xc1: sb=st.form_submit_button("Send",use_container_width=True,type="primary")
        with xc2: cb=st.form_submit_button("Clear",use_container_width=True)

    if cb:
        st.session_state.chat_history=[]; st.rerun()
    if sb and ui.strip():
        with st.spinner("Thinking…"):
            reply=(chat_with_claude(ui,st.session_state.chat_history,df) if ENHANCED_AI
                   else f"Total expenses: PKR {df[df['transaction_type']=='expense']['amount'].sum():,.0f}. Set ANTHROPIC_API_KEY for full AI answers.")
        st.session_state.chat_history.append({'role':'user','content':ui})
        st.session_state.chat_history.append({'role':'assistant','content':reply})
        st.rerun()

# ── ANALYTICS ────────────────────────────────────────────────────
elif page=="Analytics":
    page_header("Detailed Analytics","Granular spending breakdowns",eyebrow="Analysis")
    ex=df[df['transaction_type']=='expense'].copy()
    if len(ex)==0:
        empty_state("No expense data yet","Upload a statement or switch to sample data to see analytics.")
        st.stop()

    t1,t2,t3,t4 = st.tabs(["Overview","Merchants","Categories","Trends"])

    with t1:
        c1,c2=st.columns(2,gap="medium")
        with c1:
            card_open("Category Breakdown")
            ca=ex.groupby('category')['amount'].sum().sort_values(ascending=True)
            fig=go.Figure(go.Bar(x=ca.values,y=ca.index,orientation='h',marker=dict(color=[cat_color(c) for c in ca.index]),
                hovertemplate="<b>%{y}</b><br>PKR %{x:,.0f}<extra></extra>"))
            plotly_fig(fig,height=300,showlegend=False,xaxis_title="Amount (PKR)"); card_close()
        with c2:
            card_open("Spending by Day of Week")
            ex['dow']=ex['date'].dt.day_name()
            ds=ex.groupby('dow')['amount'].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
            fig=go.Figure(go.Bar(x=ds.index,y=ds.values,marker=dict(color=ACCENT,opacity=0.85),
                hovertemplate="<b>%{x}</b><br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=300,showlegend=False,yaxis_title="Total (PKR)"); card_close()

    with t2:
        card_open("Top 15 Merchants")
        tm=ex.groupby('merchant_name')['amount'].sum().nlargest(15).sort_values()
        fig=go.Figure(go.Bar(x=tm.values,y=tm.index,orientation='h',marker=dict(color=ACCENT,opacity=0.9),
            hovertemplate="<b>%{y}</b><br>PKR %{x:,.0f}<extra></extra>"))
        plotly_fig(fig,height=440,showlegend=False,xaxis_title="Total (PKR)"); card_close()
        card_open("Merchant Detail")
        md = ex.groupby('merchant_name').agg(total=('amount','sum'),count=('amount','count'),avg=('amount','mean'),
            category=('category', lambda s: s.mode().iat[0])).sort_values('total',ascending=False).head(20)
        rows = [[esc(m), cat_tag(r['category']), str(int(r['count'])), f'<span class="amt">PKR {r["avg"]:,.0f}</span>', f'<span class="amt">PKR {r["total"]:,.0f}</span>'] for m,r in md.iterrows()]
        st.markdown(html_table(['Merchant','Category','Transactions','Avg','Total'], rows, right_align={2,3,4}, max_height=380), unsafe_allow_html=True)
        card_close()

    with t3:
        card_open("Category Detail")
        cat_stats = ex.groupby('category').agg(total=('amount','sum'),count=('amount','count'),avg=('amount','mean')).sort_values('total',ascending=False)
        rows = [[cat_tag(c), str(int(r['count'])), f'<span class="amt">PKR {r["avg"]:,.0f}</span>', f'<span class="amt">PKR {r["total"]:,.0f}</span>'] for c,r in cat_stats.iterrows()]
        st.markdown(html_table(['Category','Transactions','Avg','Total'], rows, right_align={1,2,3}), unsafe_allow_html=True)
        card_close()

    with t4:
        card_open("Income vs Expenses Over Time")
        dm=df.copy(); dm['month']=dm['date'].dt.to_period('M').dt.to_timestamp()
        im2=dm[dm['transaction_type']=='income'].groupby('month')['amount'].sum()
        em2=dm[dm['transaction_type']=='expense'].groupby('month')['amount'].sum()
        am=sorted(set(im2.index)|set(em2.index))
        fig=go.Figure()
        fig.add_trace(go.Bar(name='Income',x=am,y=[im2.get(m,0) for m in am],marker_color=POSITIVE))
        fig.add_trace(go.Bar(name='Expenses',x=am,y=[em2.get(m,0) for m in am],marker_color=NEGATIVE))
        fig.update_layout(barmode='group')
        plotly_fig(fig,height=320,xaxis=dict(type='date',gridcolor='rgba(255,255,255,0.05)'),yaxis_title="PKR",hovermode='x unified')
        card_close()

# ── FRAUD DETECTION ───────────────────────────────────────────────
elif page=="Fraud Detection":
    page_header("Fraud Detection & Security","Isolation Forest anomaly detection",eyebrow="Security")
    fc2=df_fraud['is_fraud'].sum(); fp=(fc2/len(df_fraud)*100) if len(df_fraud)>0 else 0

    c1,c2=st.columns([1,2.2],gap="medium")
    with c1:
        ring = ring_svg(min(100,fp*4), size=104, stroke=9, color=NEGATIVE if fp>10 else (WARNING if fp>5 else POSITIVE))
        st.markdown(f'''<div class="side-card" style="align-items:center;">
          <div class="ring-row" style="justify-content:center;">
            <div class="ring-wrap">{ring}<div class="ring-num">{fp:.1f}%</div></div>
          </div>
          <div class="side-note" style="text-align:center;border-top:none;padding-top:0;">Fraud rate across {len(df_fraud):,} transactions</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        kpi_row([
            ("Suspicious",str(int(fc2)),"","down" if fc2>0 else None),
            ("Fraud Rate",f"{fp:.1f}%"),
            ("Safe",str(int(len(df_fraud)-fc2))),
        ])
        spacer(8)
        if fp>10:   status('danger','High fraud rate. Review immediately.')
        elif fp>5:  status('warning','Moderate fraud rate. Monitor closely.')
        else:       status('success','Fraud rate within normal range.')

    spacer(10)
    if AI_ENABLED and st.checkbox("Enable Z-Score statistical analysis"):
        with st.spinner("Analysing…"):
            an=detect_anomalies_ai(df_fraud)
            if an:
                card_open("Statistical Anomalies (Z-Score > 2.5)")
                rows = [[esc(a['merchant']), f'{a.get("z_score",0):.2f}', f'<span class="amt">PKR {a["amount"]:,.0f}</span>'] for a in an[:8]]
                st.markdown(html_table(['Merchant','Z-Score','Amount'], rows, right_align={1,2}), unsafe_allow_html=True)
                card_close()

    card_open("Flagged Transactions")
    ft=df_fraud[df_fraud['is_fraud']].copy()
    if len(ft)>0:
        if 'fraud_score' in ft.columns:
            ft['anomaly']=((ft['fraud_score'].min()-ft['fraud_score'])/(ft['fraud_score'].min()-ft['fraud_score'].max()+1e-9)*100).round(1)
        else:
            ft['anomaly']=0.0
        rows = [[r['date'].strftime('%b %d'), esc(r['merchant_name']), cat_tag(r['category']),
                 pill('High' if r['anomaly']>66 else 'Medium','danger' if r['anomaly']>66 else 'warning'),
                 f'<span class="amt">PKR {r["amount"]:,.0f}</span>'] for _,r in ft.iterrows()]
        st.markdown(html_table(['Date','Merchant','Category','Severity','Amount'], rows, right_align={4}, max_height=360), unsafe_allow_html=True)
        status('warning','Contact your bank for any unrecognized charges.')
    else:
        empty_state("No suspicious transactions","Your account currently shows no fraud indicators.")
    card_close()

    card_open("Anomaly Scatter Plot")
    fig=go.Figure()
    sd=df_fraud[~df_fraud['is_fraud']]; fd2=df_fraud[df_fraud['is_fraud']]
    fig.add_trace(go.Scatter(x=sd['date'],y=sd['amount'],mode='markers',name='Safe',
        marker=dict(color=POSITIVE,size=5,opacity=0.5),text=sd['merchant_name'],
        hovertemplate="%{text}<br>PKR %{y:,.0f}<extra></extra>"))
    if len(fd2)>0:
        fig.add_trace(go.Scatter(x=fd2['date'],y=fd2['amount'],mode='markers',name='Flagged',
            marker=dict(color=NEGATIVE,size=10,symbol='x',line=dict(width=2,color=NEGATIVE)),
            text=fd2['merchant_name'],hovertemplate="%{text}<br>PKR %{y:,.0f}<extra></extra>"))
    plotly_fig(fig,height=300,xaxis=dict(type='date',gridcolor='rgba(255,255,255,0.05)'),yaxis_title="PKR",hovermode='closest')
    card_close()

# ── RECURRING PAYMENTS ────────────────────────────────────────────
elif page=="Recurring Payments":
    page_header("Recurring Payments & Subscriptions","Auto-detected regular charges",eyebrow="Subscriptions")
    if len(recurring_df)>0:
        c1,c2=st.columns([1.6,1],gap="medium")
        with c1:
            st.markdown(hero_card("Annual Subscription Cost", f"PKR {recurring_df['total_annual'].sum():,.0f}",
                f"PKR {recurring_df['avg_amount'].sum():,.0f} / month", None,
                sorted(recurring_df['total_annual'].tolist()), f"{len(recurring_df)} active subscriptions detected"), unsafe_allow_html=True)
        with c2:
            top3 = recurring_df.nlargest(3,'total_annual')
            rows_html = "".join(f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:12.5px;"><span>{cat_tag(r["merchant_name"])}</span><span class="amt">PKR {r["total_annual"]:,.0f}</span></div>' for _,r in top3.iterrows())
            st.markdown(f'<div class="side-card"><div class="ring-meta-label">Top 3 By Annual Cost</div><div>{rows_html}</div></div>', unsafe_allow_html=True)

        spacer(16)
        card_open("Your Subscriptions","Sorted by annual cost")
        headers = ['Merchant','Category','Cycle','Times Seen','Monthly','Annual']
        has_cycle = 'billing_cycle' in recurring_df.columns
        rows = []
        for _,r in recurring_df.iterrows():
            rows.append([
                esc(r['merchant_name']), cat_tag(r['category']),
                r.get('billing_cycle','Monthly') if has_cycle else 'Monthly',
                str(int(r['count'])),
                f'<span class="amt">PKR {r["avg_amount"]:,.0f}</span>',
                f'<span class="amt">PKR {r["total_annual"]:,.0f}</span>',
            ])
        st.markdown(html_table(headers, rows, right_align={3,4,5}), unsafe_allow_html=True)
        status('warning',f'Annual subscription burden: <strong>PKR {recurring_df["total_annual"].sum():,.0f}</strong>')
        card_close()

        card_open("Cost Distribution")
        fig=go.Figure(data=[go.Pie(labels=recurring_df['merchant_name'],values=recurring_df['total_annual'],hole=0.6,
            marker=dict(colors=[cat_color(m) for m in recurring_df['merchant_name']],line=dict(color=SURFACE1,width=2)))])
        plotly_fig(fig,height=260,showlegend=True,legend=dict(orientation='v',x=1.0,y=0.5,font=dict(size=10)))
        card_close()

        if AI_ENABLED:
            card_open("AI Subscription Optimisation")
            if st.button("Analyse with AI"):
                with st.spinner("…"):
                    aa=analyze_subscriptions_ai(recurring_df)
                    if '_savings_opportunities' in aa: status('success',f'Savings: {aa["_savings_opportunities"]}')
                    if '_alternatives' in aa: status('info',f'Alternatives: {aa["_alternatives"]}')
            card_close()
    else:
        empty_state("No recurring payments detected","Upload multiple months of data so patterns can be identified.")

# ── FINANCIAL HEALTH ──────────────────────────────────────────────
elif page=="Financial Health":
    page_header("Financial Health Score","Composite score 0–100 across 5 dimensions",eyebrow="Health")
    ti2=df[df['transaction_type']=='income']['amount'].sum()
    ts2=df[df['transaction_type']=='expense']['amount'].sum()
    health=(compute_financial_health_score(df,ti2,ts2,recurring_df) if ENHANCED_AI
            else generate_financial_health_assessment(df,ti2,ts2))
    if not ENHANCED_AI:
        health.setdefault('total_score',int(str(health.get('_health_score','0/100')).split('/')[0]))
        health.setdefault('grade','A' if health['total_score']>=85 else 'B' if health['total_score']>=70 else 'C')
        health.setdefault('recommendations',[health.get('_recommendations','')])
        health.setdefault('sub_scores',{})

    sc2=int(health.get('total_score',0)); gr=health.get('grade','?')
    col=(POSITIVE if sc2>=75 else WARNING if sc2>=50 else NEGATIVE)

    x1,x2=st.columns([1,2],gap="large")
    with x1:
        ring = ring_svg(sc2, size=132, stroke=11, color=col)
        st.markdown(f'''<div style="text-align:center;padding:20px;background:var(--surface-1);border:1px solid var(--border);border-radius:var(--radius);">
          <div style="position:relative;display:inline-block;">{ring}
            <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
              <div style="font-family:var(--font-mono);font-size:30px;font-weight:700;color:{col};line-height:1;">{sc2}</div>
              <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-faint);">/ 100</div>
            </div>
          </div>
          <div style="font-size:22px;font-weight:800;color:{col};margin-top:12px;">Grade {gr}</div>
        </div>''',unsafe_allow_html=True)
    with x2:
        if health.get('sub_scores'):
            card_open("Score Breakdown")
            for k,(lbl,mx) in [('savings',('Savings Ratio',30)),('consistency',('Consistency',20)),
                ('subscription',('Subscription Load',20)),('risk',('Risk Profile',15)),('income_stability',('Income Stability',15))]:
                v=health['sub_scores'].get(k,0)
                st.markdown(f"**{lbl}** — {v:.1f}/{mx}")
                st.progress(min(1.0,v/mx))
            card_close()
        else:
            spacer(4)

    spacer(14)
    card_open("Recommendations")
    for r in health.get('recommendations',[]):
        if r: status('info',r)
    sr2=health.get('savings_rate',0)
    if sr2>=20: status('success',f"Strong savings rate of {sr2:.1f}%.")
    elif sr2>0: status('warning',f"Savings rate {sr2:.1f}% — aim for 20%+.")
    card_close()

# ── SMART ALERTS ──────────────────────────────────────────────────
elif page=="Smart Alerts":
    page_header("Smart Alerts","Proactive financial notifications",eyebrow="Monitoring")
    if not ENHANCED_AI:
        status('warning','Enhanced AI module required.'); st.stop()
    alerts=generate_smart_alerts(df,recurring_df,df_fraud)
    da=[a for a in alerts if a['type']=='danger']; wa=[a for a in alerts if a['type']=='warning']; sa=[a for a in alerts if a['type']=='success']
    kpi_row([
        ("Critical",str(len(da)),"","down" if da else None),
        ("Warnings",str(len(wa))),
        ("All Clear",str(len(sa)),"","up" if sa else None),
    ])
    spacer(10)
    card_open("Active Alerts")
    for a in alerts:
        act=f' <span style="opacity:.65;font-family:var(--font-mono);font-size:11px;">→ {a["action"]}</span>' if a['action'] else ''
        status(a['type'],f'<strong>{a["title"]}:</strong> {a["message"]}{act}')
    card_close()

# ── BUDGET PLANNING ───────────────────────────────────────────────
elif page=="Budget Planning":
    page_header("Budget Planning","Compare actual vs recommended spend",eyebrow="Planning")
    ex3=df[df['transaction_type']=='expense'].copy()
    ti3=df[df['transaction_type']=='income']['amount'].sum(); te3=ex3['amount'].sum()
    if ti3>0:
        ms=max(1,(df['date'].max()-df['date'].min()).days/30)
        ami=ti3/ms; ame=te3/ms; sr3=(ti3-te3)/ti3*100
        kpi_row([
            ("Avg Monthly Income",f"PKR {ami:,.0f}"),
            ("Avg Monthly Spend",f"PKR {ame:,.0f}"),
            ("Savings Rate",f"{sr3:.1f}%","","up" if sr3>=15 else "down"),
        ])
        spacer()
        std={'Housing/Rent':0.25,'Utilities':0.08,'Groceries':0.12,'Transport':0.12,'Food & Dining':0.10,'Healthcare':0.06,'Entertainment & Shopping':0.15,'Subscriptions':0.04,'Savings':0.08}

        t1,t2,t3 = st.tabs(["Recommended Budget","Actual Spending","Recommendations"])
        with t1:
            rows = [[c, f"{p*100:.0f}%", f'<span class="amt">PKR {ami*p:,.0f}</span>'] for c,p in std.items()]
            st.markdown(html_table(['Category','Recommended %','Monthly'], rows, right_align={1,2}), unsafe_allow_html=True)
        with t2:
            ac3=ex3.groupby('category')['amount'].agg(['sum','count','mean']).round(0)
            ac3.columns=['Total','Txns','Avg']; ac3['pct']=(ac3['Total']/ti3*100).round(1)
            rows = [[cat_tag(c), str(int(r['Txns'])), f'<span class="amt">PKR {r["Avg"]:,.0f}</span>', f'<span class="amt">PKR {r["Total"]:,.0f}</span>', f'{r["pct"]:.1f}%'] for c,r in ac3.sort_values('Total',ascending=False).iterrows()]
            st.markdown(html_table(['Category','Txns','Avg','Total','% of Income'], rows, right_align={1,2,3,4}), unsafe_allow_html=True)
        with t3:
            rm={'Rent':'Housing/Rent','Utilities':'Utilities','Groceries':'Groceries','Transportation':'Transport',
                'Food & Dining':'Food & Dining','Healthcare':'Healthcare','Entertainment':'Entertainment & Shopping',
                'Shopping':'Entertainment & Shopping','Subscriptions':'Subscriptions'}
            ac3b=ex3.groupby('category')['amount'].sum()
            found=False
            for ac4,rc4 in rm.items():
                if ac4 not in ac3b.index: continue
                rm2=ami*std.get(rc4,0.1); am2=ac3b.loc[ac4]/ms
                if am2>rm2*1.2: status('warning',f'<strong>{ac4}:</strong> PKR {am2:,.0f}/mo — recommended PKR {rm2:,.0f}/mo'); found=True
            if not found: status('success','Spending aligns with recommended allocations.')
            if AI_ENABLED:
                spacer(6)
                if st.button("Generate AI Budget Recommendations"):
                    with st.spinner("…"):
                        ar=generate_budget_recommendations(df,ti3)
                        if '_ai_tips' in ar: status('success',ar['_ai_tips'])
    else:
        status('warning','No income transactions found.')

# ── MONTHLY REPORT ────────────────────────────────────────────────
elif page=="Monthly Report":
    page_header("Monthly Financial Report","Detailed breakdown by month",eyebrow="Reports")
    months=sorted(df['date'].dt.to_period('M').unique(),reverse=True)
    sm=st.selectbox("Month",months,key="msel")
    mdf=df[df['date'].dt.to_period('M')==sm]
    mi=mdf[mdf['transaction_type']=='income']['amount'].sum(); me=mdf[mdf['transaction_type']=='expense']['amount'].sum()
    mb=mi-me; msr=(mb/mi*100) if mi>0 else 0
    kpi_row([
        ("Income",f"PKR {mi:,.0f}"),
        ("Expenses",f"PKR {me:,.0f}"),
        ("Balance",f"PKR {mb:,.0f}","","up" if mb>=0 else "down"),
        ("Savings Rate",f"{msr:.1f}%"),
    ])
    spacer()
    x1,x2=st.columns(2,gap="medium")
    with x1:
        card_open(f"Categories — {sm}")
        cm=mdf[mdf['transaction_type']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
        if len(cm)>0:
            fig=go.Figure(go.Bar(x=cm.index,y=cm.values,marker=dict(color=[cat_color(c) for c in cm.index]),hovertemplate="<b>%{x}</b><br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=240,showlegend=False,yaxis_title="PKR")
        card_close()
    with x2:
        card_open(f"Daily Spending — {sm}")
        dl=mdf[mdf['transaction_type']=='expense'].groupby(mdf['date'].dt.day)['amount'].sum()
        if len(dl)>0:
            fig=go.Figure(go.Scatter(x=dl.index,y=dl.values,mode='lines+markers',fill='tozeroy',fillcolor='rgba(139,124,246,.10)',line=dict(color=ACCENT,width=2),marker=dict(size=5),hovertemplate="Day %{x}<br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=240,showlegend=False,xaxis_title="Day",yaxis_title="PKR")
        card_close()
    card_open("Transactions")
    td = mdf.sort_values('date',ascending=False)
    rows = [[r['date'].strftime('%b %d'), esc(r['merchant_name']), cat_tag(r['category']),
             pill('Income','positive') if r['transaction_type']=='income' else pill('Expense','neutral'),
             amount_cell(r['amount'], r['transaction_type'])] for _,r in td.iterrows()]
    st.markdown(html_table(['Date','Merchant','Category','Type','Amount'], rows, right_align={4}, max_height=420), unsafe_allow_html=True)
    card_close()

# ── BILL REMINDERS ────────────────────────────────────────────────
elif page=="Bill Reminders":
    page_header("Bill Reminders","Upcoming bills from recurring payment detection",eyebrow="Payments")
    if len(recurring_df)>0:
        today=pd.Timestamp.now()
        kpi_row([
            ("Monthly Total",f"PKR {recurring_df['avg_amount'].sum():,.0f}"),
            ("Annual Total",f"PKR {recurring_df['total_annual'].sum():,.0f}"),
            ("Bills",str(len(recurring_df))),
        ])
        spacer(10)
        card_open("Upcoming Bills (Estimated)")
        bills = recurring_df.sort_values('avg_amount',ascending=False)
        rows = []
        for _,r in bills.iterrows():
            due = (today+timedelta(days=30))
            days_until = (due-today).days
            st_kind = 'danger' if days_until<=0 else ('warning' if days_until<=3 else 'positive')
            st_lbl = 'Due Now' if days_until<=0 else ('Due Soon' if days_until<=3 else 'Upcoming')
            rows.append([cat_tag(r['merchant_name']), esc(r['category']), f'<span class="amt">PKR {r["avg_amount"]:,.0f}</span>',
                         due.strftime('%b %d, %Y'), pill(st_lbl, st_kind)])
        st.markdown(html_table(['Bill','Category','Amount','Est. Due','Status'], rows, right_align={2}), unsafe_allow_html=True)
        card_close()
        if AI_ENABLED:
            card_open("AI Bill Tips")
            if st.button("Get AI Bill Optimisation"):
                with st.spinner("…"):
                    r=generate_smart_bill_reminders(recurring_df,df)
                    if '_recommendations' in r: status('info',r['_recommendations'])
                    if '_urgency' in r: status('warning',r['_urgency'])
            card_close()
    else:
        empty_state("No recurring bills detected","Upload multiple months of data to see bills.")

# ── OCR RECEIPT UPLOAD ────────────────────────────────────────────
elif page=="OCR Receipt Upload":
    page_header("OCR Receipt / Screenshot Upload","Extract transaction details from any receipt photo",eyebrow="Capture")

    CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY","")

    try:
        import pytesseract
        import os as _os
        for tp in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                   r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']:
            if _os.path.exists(tp): pytesseract.pytesseract.tesseract_cmd = tp; break
        pytesseract.get_tesseract_version()
        TESS_OK = True
    except Exception:
        TESS_OK = False

    if CLAUDE_KEY:
        status('success','Claude Vision AI active — reads any receipt, screenshot or slip with high accuracy.')
    elif TESS_OK:
        status('success','Tesseract OCR is installed and ready. Upload a receipt image below.')
    else:
        status('warning',
            'No OCR engine detected. <strong>Two options:</strong><br>'
            '1. <strong>Claude Vision (recommended, no install):</strong> add <code>ANTHROPIC_API_KEY=sk-ant-...</code> to your .env file<br>'
            '2. <strong>Tesseract (local, free):</strong> run in terminal as Administrator: '
            '<code>winget install UB-Mannheim.TesseractOCR</code> → restart VS Code → <code>pip install pytesseract Pillow</code>')

    spacer(8)

    card_open("Upload Receipt or Screenshot","Supported: JPG, PNG, WEBP, BMP — any bank, wallet, or payment app")
    img_file = st.file_uploader("ocr_up", type=['jpg','jpeg','png','webp','bmp'], label_visibility="collapsed")
    card_close()

    def run_ocr(img_bytes, mime):
        if CLAUDE_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=CLAUDE_KEY)
                b64 = base64.b64encode(img_bytes).decode()
                resp = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400,
                    messages=[{"role":"user","content":[
                        {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}},
                        {"type":"text","text":
                         "This is a bank/wallet payment receipt. Extract:\n"
                         "MERCHANT: (who received money or service name)\n"
                         "AMOUNT: (number only, e.g. 170)\n"
                         "DATE: (e.g. May 12 2026)\n"
                         "TXN_ID: (transaction/reference ID if visible)\n"
                         "FROM: (sender name if visible)\n"
                         "TO: (recipient name if visible)\n"
                         "TYPE: (Transfer/Payment/Bill etc)\n"
                         "Write UNKNOWN for fields not visible."}
                    ]}])
                return resp.content[0].text, "Claude Vision AI"
            except Exception as e:
                pass

        try:
            from PIL import Image
            import pytesseract, io as _io
            for tp in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                       r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                       '/usr/bin/tesseract','/usr/local/bin/tesseract']:
                import os as _os
                if _os.path.exists(tp): pytesseract.pytesseract.tesseract_cmd = tp; break
            img = Image.open(_io.BytesIO(img_bytes))
            w, h = img.size
            if w < 800: img = img.resize((w*2, h*2), Image.LANCZOS)
            text = pytesseract.image_to_string(img, config='--psm 6')
            return text.strip(), "Tesseract OCR"
        except ImportError:
            return "TESSERACT_NOT_INSTALLED", "none"
        except Exception as e:
            return f"OCR_ERROR: {e}", "none"

    def parse_ocr_result(text, method):
        r = {'merchant':None,'amount':None,'date':None,'txn_id':None,'from':None,'to':None,'type':None}

        if method == "Claude Vision AI":
            for line in text.split('\n'):
                line = line.strip()
                for key in ['MERCHANT','AMOUNT','DATE','TXN_ID','FROM','TO','TYPE']:
                    if line.upper().startswith(key+':'):
                        v = line.split(':',1)[1].strip()
                        if v.upper() not in ('UNKNOWN','N/A','NONE',''):
                            k2 = key.lower()
                            if k2 == 'amount':
                                try: r['amount'] = float(re.sub(r'[^\d.]','',v))
                                except: pass
                            else: r[k2] = v
        else:
            lines = text.split('\n')
            amt_pats = [
                re.compile(r'Rs\.?\s*([\d,]+\.?\d{0,2})', re.I),
                re.compile(r'PKR\s*([\d,]+\.?\d{0,2})', re.I),
                re.compile(r'^([\d,]+\.\d{2})$'),
            ]
            date_pat = re.compile(r'(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})', re.I)
            tid_pat  = re.compile(r'(?:TID|Transaction\s*ID|Ref|TxID)[:\s#]+([A-Za-z0-9]{6,})', re.I)
            to_pat   = re.compile(r'(?:Transferred\s+to|To)[:\s]+(.+)', re.I)
            from_pat = re.compile(r'From[:\s]+(.+)', re.I)

            for line in lines:
                line_s = line.strip()
                if not r['amount']:
                    for p in amt_pats:
                        m = p.search(line_s)
                        if m:
                            try: r['amount'] = float(m.group(1).replace(',','')); break
                            except: pass
                if not r['date']:
                    dm = date_pat.search(line_s)
                    if dm: r['date'] = dm.group(1)
                if not r['txn_id']:
                    tm = tid_pat.search(line_s)
                    if tm: r['txn_id'] = tm.group(1)
                if not r['to']:
                    tom = to_pat.search(line_s)
                    if tom: r['to'] = tom.group(1).strip()[:50]
                if not r['from']:
                    fm = from_pat.search(line_s)
                    if fm: r['from'] = fm.group(1).strip()[:50]

            for svc in ['JazzCash','NayaPay','Easypaisa','HBL','Meezan','Foodpanda','Careem','Daraz','Netflix']:
                if svc.lower() in text.lower():
                    r['merchant'] = svc; break

        return r

    if img_file:
        raw = img_file.read()
        mime = f"image/{img_file.type.split('/')[-1]}" if img_file.type else "image/jpeg"

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(raw, caption="Uploaded image", use_column_width=True)

        with c2:
            with st.spinner("Extracting transaction data…"):
                txt, method_used = run_ocr(raw, mime)

            if txt in ("TESSERACT_NOT_INSTALLED",) or txt.startswith("OCR_ERROR"):
                status('danger',
                    f'<strong>OCR not available.</strong> {txt}<br><br>'
                    'Fix: run as Administrator: <code>winget install UB-Mannheim.TesseractOCR</code><br>'
                    'Or set ANTHROPIC_API_KEY to use Claude Vision (no install needed).')
            else:
                status('info', f'Extracted using: <strong>{method_used}</strong>')
                card_open("Raw Extracted Text")
                st.text_area("raw_ocr", txt, height=160, label_visibility="collapsed")
                card_close()

                ext = parse_ocr_result(txt, method_used)
                if ext.get('amount'):
                    card_open("Transaction Detected")
                    xa, xb = st.columns(2)
                    with xa:
                        st.metric("Amount",   f"PKR {ext['amount']:,.0f}")
                        st.metric("Merchant", ext.get('merchant') or ext.get('to') or 'Unknown')
                    with xb:
                        st.metric("Date",     ext.get('date') or 'Unknown')
                        st.metric("Type",     ext.get('type') or 'Transfer/Payment')
                    if ext.get('txn_id'): st.caption(f"TID: {ext['txn_id']}")
                    if ext.get('from'):   st.caption(f"From: {ext['from']}")
                    if ext.get('to'):     st.caption(f"To: {ext['to']}")

                    st.write("**Confirm before adding:**")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    with col_e1:
                        confirm_merchant = st.text_input("Merchant", value=ext.get('merchant') or ext.get('to') or 'Unknown', key="ocr_merchant")
                    with col_e2:
                        confirm_amount = st.number_input("Amount (PKR)", value=float(ext['amount']), min_value=0.0, key="ocr_amount")
                    with col_e3:
                        confirm_type = st.selectbox("Type", ["expense","income"], key="ocr_txn_type")

                    confirm_date = st.date_input("Date", value=datetime.now().date(), key="ocr_date")
                    confirm_cat  = st.selectbox("Category", ["Financial","Food & Dining","Utilities","Transportation","Shopping","Entertainment","Healthcare","Other"], key="ocr_cat")

                    if st.button("Add to My Finances", use_container_width=True, type="primary"):
                        new_row = {
                            'date':             pd.Timestamp(confirm_date),
                            'merchant_name':    confirm_merchant,
                            'category':         confirm_cat,
                            'amount':           abs(confirm_amount),
                            'description':      f'OCR: {confirm_merchant}',
                            'transaction_type': confirm_type,
                            'transaction_id':   f'OCR{len(st.session_state.ocr_transactions):04d}',
                        }
                        st.session_state.ocr_transactions.append(new_row)
                        status('success', f'Added PKR {confirm_amount:,.0f} from {confirm_merchant}. Go to Dashboard to see it reflected.')

                    if st.session_state.ocr_transactions:
                        st.caption(f"{len(st.session_state.ocr_transactions)} OCR transaction(s) added to your finances this session.")
                        if st.button("Clear OCR Transactions", key="clear_ocr"):
                            st.session_state.ocr_transactions = []
                            st.rerun()
                    card_close()
                else:
                    status('warning',
                        'Amount not detected. Tips:<br>'
                        '– Ensure the image is sharp and well-lit<br>'
                        '– Try a screenshot instead of a photo<br>'
                        '– Use Claude Vision (ANTHROPIC_API_KEY) for best results')


elif page=="Reports & Export":
    page_header("Reports & Export","Download financial reports",eyebrow="Export")

    t1,t2,t3,t4 = st.tabs(["Monthly Summary","Annual Summary","Category Analysis","Subscription Audit"])
    ef = st.radio("Format", ["Excel (XLSX)","CSV"], horizontal=True, key="export_fmt")
    spacer(6)

    with t1:
        months=sorted(df['date'].dt.to_period('M').unique(),reverse=True)
        sel=st.multiselect("Months",months,default=[months[0]] if months else [])
        if sel:
            rows=[]
            for m in sel:
                mdf=df[df['date'].dt.to_period('M')==m]
                inc=mdf[mdf['transaction_type']=='income']['amount'].sum()
                exp=mdf[mdf['transaction_type']=='expense']['amount'].sum()
                rows.append({'Month':str(m),'Income':f"PKR {inc:,.0f}",'Expenses':f"PKR {exp:,.0f}",'Balance':f"PKR {inc-exp:,.0f}",'Transactions':len(mdf)})
            rdf=pd.DataFrame(rows); st.dataframe(rdf,use_container_width=True,hide_index=True)
            if ef=="Excel (XLSX)":
                _buf=io.BytesIO()
                with pd.ExcelWriter(_buf,engine='openpyxl') as _w: rdf.to_excel(_w,index=False)
                st.download_button("Download Excel",_buf.getvalue(),f"monthly_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.download_button("Download CSV",rdf.to_csv(index=False),f"monthly_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    with t2:
        ai2=df[df['transaction_type']=='income']['amount'].sum(); ae2=df[df['transaction_type']=='expense']['amount'].sum()
        sv2=ai2-ae2; sr4=(sv2/ai2*100) if ai2>0 else 0; pd2=(df['date'].max()-df['date'].min()).days; pm=max(1,pd2/30)
        sdf=pd.DataFrame({'Metric':['Total Income','Total Expenses','Net Savings','Savings Rate','Avg Monthly Income','Avg Monthly Expenses','Days Tracked'],
            'Amount':[f"PKR {ai2:,.0f}",f"PKR {ae2:,.0f}",f"PKR {sv2:,.0f}",f"{sr4:.1f}%",f"PKR {ai2/pm:,.0f}",f"PKR {ae2/pm:,.0f}",str(pd2)]})
        st.dataframe(sdf,use_container_width=True,hide_index=True)
        if ef=="Excel (XLSX)":
            _buf=io.BytesIO()
            with pd.ExcelWriter(_buf,engine='openpyxl') as _w:
                sdf.to_excel(_w,sheet_name='Annual',index=False)
                cs=df[df['transaction_type']=='expense'].groupby('category')['amount'].agg(['sum','count','mean']).reset_index()
                cs.columns=['Category','Total','Count','Avg']; cs.to_excel(_w,sheet_name='By Category',index=False)
            st.download_button("Download Excel",_buf.getvalue(),f"annual_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.download_button("Download CSV",sdf.to_csv(index=False),f"annual_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    with t3:
        ca5=df[df['transaction_type']=='expense'].groupby('category').agg({'amount':['sum','count','mean','min','max']}).round(0)
        ca5.columns=['Total (PKR)','Count','Avg (PKR)','Min (PKR)','Max (PKR)']; ca5=ca5.sort_values('Total (PKR)',ascending=False)
        st.dataframe(ca5,use_container_width=True)
        if ef=="Excel (XLSX)":
            _buf=io.BytesIO()
            with pd.ExcelWriter(_buf,engine='openpyxl') as _w: ca5.to_excel(_w)
            st.download_button("Download Excel",_buf.getvalue(),f"categories_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.download_button("Download CSV",ca5.to_csv(),f"categories_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    with t4:
        if len(recurring_df)>0:
            au=recurring_df.copy().sort_values('total_annual',ascending=False)
            kpi_row([
                ("Subscriptions",str(len(au))),
                ("Annual PKR",f"PKR {au['total_annual'].sum():,.0f}"),
                ("Monthly PKR",f"PKR {au['total_annual'].sum()/12:,.0f}"),
            ])
            spacer(10)
            dc2=['merchant_name','category','avg_amount','count','total_annual']
            dn2=['Subscription','Category','Monthly (PKR)','Count','Annual (PKR)']
            ad=au[dc2].copy(); ad.columns=dn2; st.dataframe(ad,use_container_width=True,hide_index=True)
            au['u']=(au['count']/au['count'].max()*100).round(0)
            un=au[au['u']<50]
            if len(un)>0: status('warning',f'<strong>{len(un)}</strong> underused subscription(s) — consider cancelling.')
            else: status('success','All subscriptions actively used.')
            if ef=="Excel (XLSX)":
                _buf=io.BytesIO()
                with pd.ExcelWriter(_buf,engine='openpyxl') as _w: ad.to_excel(_w,sheet_name='Subscriptions',index=False)
                st.download_button("Download Excel",_buf.getvalue(),f"subscriptions_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.download_button("Download CSV",ad.to_csv(index=False),f"subscriptions_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")
        else:
            empty_state("No subscriptions detected","Upload recurring transaction history to see a subscription audit.")
