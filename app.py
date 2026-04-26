import streamlit as st
import cv2
import numpy as np
import subprocess
import os
import re
import tempfile
import zipfile
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import io
import base64

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectoLine — Image Vectorizer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state init — must happen before sidebar rendering ────────────────
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:        #0e0e11;
    --bg2:       #17171c;
    --bg3:       #1e1e26;
    --accent:    #7cfc8e;
    --accent2:   #4af0c8;
    --accent3:   #ffbb55;
    --muted:     #7a7a99;
    --text:      #e0e0f0;
    --text-sec:  #b8b8d0;
    --border:    #2a2a38;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0e0e11 0%, #17171c 50%, #1a1a24 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,252,142,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(74,240,200,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0 0 0.3rem 0;
    line-height: 1;
}
.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-sec);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 1rem 0;
}
.pipeline-badge {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    letter-spacing: 1px;
}
.pipeline-badge span { color: var(--muted); margin: 0 6px; }

/* ── Pipeline Settings (always visible) ── */
.pipeline-settings-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.pipeline-settings-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    border-left: 3px solid var(--accent);
    padding-left: 8px;
}

/* ── Mode Cards ── */
.mode-grid { display: flex; gap: 10px; margin-top: 0.85rem; flex-wrap: wrap; }
.mode-card {
    flex: 1; min-width: 160px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.82rem;
    color: var(--text);
}
.mode-card strong {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    display: block;
    margin-bottom: 4px;
}
.mode-card .tag {
    display: inline-block;
    background: rgba(124,252,142,0.1);
    border: 1px solid rgba(124,252,142,0.2);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.68rem;
    color: var(--accent);
    margin-top: 4px;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 12px; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-left: 3px solid var(--accent);
    padding-left: 8px;
}

.img-panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

/* ── Widget Overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] > div {
    background-color: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
label, .stSlider label, .stSelectbox label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}
p, li { color: var(--text); }

[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label { color: var(--text) !important; }

.stButton > button,
.stButton > button:focus,
.stButton > button:active,
.stButton > button:visited,
div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button:focus,
div[data-testid="stBaseButton-secondary"],
div[data-testid="stBaseButton-secondary"]:focus {
    background: var(--accent) !important;
    background-color: var(--accent) !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
}
/* Force ALL children of any button to be white */
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div,
div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button span,
div[data-testid="stButton"] > button div,
div[data-testid="stBaseButton-secondary"] p,
div[data-testid="stBaseButton-secondary"] span {
    color: #ffffff !important;
    fill: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
div[data-testid="stBaseButton-secondary"]:hover {
    background: #5dd96e !important;
    background-color: #5dd96e !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}
.stButton > button:hover *,
div[data-testid="stButton"] > button:hover * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
}
[data-testid="stDownloadButton"] > button:hover { background: rgba(74,240,200,0.08) !important; }

.stAlert { background: var(--bg3) !important; border-color: var(--border) !important; border-radius: 10px !important; }
[data-testid="stProgress"] > div > div { background-color: var(--accent) !important; }

[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-sec) !important;
    letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

hr { border-color: var(--border) !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
header[data-testid="stHeader"] > div:first-child { visibility: hidden; }

[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    color: var(--accent) !important;
}
[data-testid="collapsedControl"] svg path { fill: var(--accent) !important; }
[data-testid="stToggle"] label { color: var(--text) !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def image_to_line_art(img_array, mode='canny', threshold=128):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if mode == 'canny':
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, otsu * 0.5, otsu)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        return cv2.bitwise_not(edges)
    elif mode == 'adaptive':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif mode == 'xdog':
        g1 = cv2.GaussianBlur(gray.astype(float), (0, 0), 1.4)
        g2 = cv2.GaussianBlur(gray.astype(float), (0, 0), 1.4 * 1.6)
        xdog = np.where((g1 - g2) >= 0.01, 1.0, 1.0 + np.tanh(20.0 * (g1 - g2)))
        xdog = (xdog * 255).clip(0, 255).astype(np.uint8)
        _, result = cv2.threshold(xdog, threshold, 255, cv2.THRESH_BINARY)
        return result
    raise ValueError(f"Unknown mode: {mode}")


def line_art_to_svg_adv(line_art, out_svg, turdsize=2, alphamax=1.0, opttol=0.2):
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as f:
        pbm = f.name
    try:
        Image.fromarray(line_art).convert('1').save(pbm)
        r = subprocess.run(
            ['potrace', pbm, '--svg', '-o', out_svg,
             '--turdsize', str(turdsize),
             '--alphamax', str(alphamax),
             '--opttolerance', str(opttol)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
    finally:
        if os.path.exists(pbm):
            os.remove(pbm)


def _tokenize(d):
    return re.findall(r'[MmCcLlZzHhVvSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)

def _cubic(p0, p1, p2, p3, n=8):
    t = np.linspace(0, 1, n + 1); u = 1 - t
    x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
    y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
    return list(zip(x.tolist(), y.tolist()))

def _rdp(pts, eps):
    if len(pts) < 3: return pts
    arr  = np.array(pts)
    keep = np.zeros(len(arr), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(arr) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2: continue
        seg = arr[e] - arr[s]; seg_len = np.linalg.norm(seg)
        if seg_len < 1e-10:
            dists = np.linalg.norm(arr[s+1:e] - arr[s], axis=1)
        else:
            u = seg / seg_len; vecs = arr[s+1:e] - arr[s]
            proj = np.clip(vecs @ u, 0, seg_len)
            closest = arr[s] + np.outer(proj, u)
            dists = np.linalg.norm(arr[s+1:e] - closest, axis=1)
        idx = int(np.argmax(dists))
        if dists[idx] > eps:
            mid = s + 1 + idx; keep[mid] = True
            stack.append((s, mid)); stack.append((mid, e))
    return arr[keep].tolist()

def _parse_d_to_subpaths(d):
    tokens = _tokenize(d)
    if not tokens: return []
    tok_cmds = [(i, t) for i, t in enumerate(tokens) if re.match(r'^[A-Za-z]$', t)]
    blocks = []
    for ci, (ti, cmd_char) in enumerate(tok_cmds):
        next_ti = tok_cmds[ci+1][0] if ci+1 < len(tok_cmds) else len(tokens)
        nums_for_cmd = [float(t) for t in tokens[ti+1:next_ti]
                        if re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)', t)]
        blocks.append((cmd_char, nums_for_cmd))
    subpaths = []; cur = [0.0, 0.0]; sub_pts = []; closed = False
    for cmd_char, nums in blocks:
        if cmd_char == 'M':
            if sub_pts: subpaths.append((sub_pts, closed))
            cur = [nums[0], nums[1]]; sub_pts = [tuple(cur)]; closed = False
            j = 2
            while j + 1 <= len(nums) - 1: cur = [nums[j], nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char == 'm':
            if sub_pts: subpaths.append((sub_pts, closed))
            cur = [cur[0]+nums[0], cur[1]+nums[1]]; sub_pts = [tuple(cur)]; closed = False
            j = 2
            while j + 1 <= len(nums) - 1: cur = [cur[0]+nums[j], cur[1]+nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char == 'C':
            j = 0
            while j + 6 <= len(nums):
                p0=tuple(cur); p1=(nums[j],nums[j+1]); p2=(nums[j+2],nums[j+3]); p3=(nums[j+4],nums[j+5])
                sub_pts.extend(_cubic(p0,p1,p2,p3,n=8)[1:]); cur=list(p3); j+=6
        elif cmd_char == 'c':
            j = 0
            while j + 6 <= len(nums):
                p0=tuple(cur); p1=(cur[0]+nums[j],cur[1]+nums[j+1]); p2=(cur[0]+nums[j+2],cur[1]+nums[j+3]); p3=(cur[0]+nums[j+4],cur[1]+nums[j+5])
                sub_pts.extend(_cubic(p0,p1,p2,p3,n=8)[1:]); cur=list(p3); j+=6
        elif cmd_char == 'L':
            j = 0
            while j + 2 <= len(nums): cur=[nums[j],nums[j+1]]; sub_pts.append(tuple(cur)); j+=2
        elif cmd_char == 'l':
            j = 0
            while j + 2 <= len(nums): cur=[cur[0]+nums[j],cur[1]+nums[j+1]]; sub_pts.append(tuple(cur)); j+=2
        elif cmd_char in ('Z','z'): closed = True
    if sub_pts: subpaths.append((sub_pts, closed))
    return subpaths

def _simplify_d(d, eps):
    subpaths = _parse_d_to_subpaths(d)
    if not subpaths: return d, 0, 0
    out_parts = []; pts_before = 0; pts_after = 0
    for pts, closed in subpaths:
        pts_before += len(pts)
        if len(pts) < 2: continue
        simplified = _rdp(pts, eps); pts_after += len(simplified)
        if len(simplified) < 2: continue
        s = simplified
        part = f"M {s[0][0]:.2f},{s[0][1]:.2f} L " + ' '.join(f'{x:.2f},{y:.2f}' for x,y in s[1:])
        if closed: part += ' Z'
        out_parts.append(part)
    return (' '.join(out_parts) if out_parts else d), pts_before, pts_after

def simplify_svg_content(content, epsilon=1.0):
    total_before = 0; total_after = 0
    def _replace(m):
        nonlocal total_before, total_after
        simplified, pb, pa = _simplify_d(m.group(1), epsilon)
        total_before += pb; total_after += pa
        return f'd="{simplified}"'
    result = re.sub(r'd="([^"]+)"', _replace, content)
    return result, total_before, total_after


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
if st.session_state.get("sidebar_open", True):
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 0.8rem 0;">
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                        color:#7cfc8e;letter-spacing:3px;text-transform:uppercase;
                        margin-bottom:0.3rem;">✦ VectoLine</div>
            <div style="font-size:0.78rem;color:#e0e0f0;">Advanced Settings & Reference</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # ── Features Summary ──
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:#ffbb55;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.9rem;">★ Features</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#e0e0f0;line-height:1.85;">
        <div style="margin-bottom:0.7rem;">
            <span style="color:#7cfc8e;font-weight:700;">🖼 Single Image</span><br>
            <span style="color:#b8b8d0;">Upload PNG / JPG / BMP and convert to a clean SVG vector. Outputs line art preview + downloadable SVG.</span>
        </div>
        <div style="margin-bottom:0.7rem;">
            <span style="color:#4af0c8;font-weight:700;">🔬 Mode Compare</span><br>
            <span style="color:#b8b8d0;">Run all 3 line-art modes side-by-side to pick the best look before vectorizing.</span>
        </div>
        <div style="margin-bottom:0.7rem;">
            <span style="color:#ffbb55;font-weight:700;">📄 PDF Batch</span><br>
            <span style="color:#b8b8d0;">Extract every image from a PDF and vectorize all at once. Download as a ZIP of SVGs.</span>
        </div>
        <div style="margin-bottom:0.7rem;">
            <span style="color:#7cfc8e;font-weight:700;">⚡ RDP Simplification</span><br>
            <span style="color:#b8b8d0;">Ramer-Douglas-Peucker reduces path points, shrinking file size while preserving shape.</span>
        </div>
        <div>
            <span style="color:#4af0c8;font-weight:700;">📊 Live Metrics</span><br>
            <span style="color:#b8b8d0;">See SVG size, path count, and point-reduction % after every conversion.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:#7cfc8e;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.7rem;">⚙ Advanced Potrace</div>""", unsafe_allow_html=True)
        turdsize = st.slider("Turd size (noise removal)", 0, 20, 2, 1,
                             help="Suppress speckles smaller than N pixels")
        alphamax = st.slider("Alpha max (corner smoothing)", 0.0, 1.5, 1.0, 0.1,
                             help="0 = sharp corners · 1.5 = very smooth")
        opttol   = st.slider("Opt tolerance (curve fit)", 0.1, 1.0, 0.2, 0.1,
                             help="Tolerance for Bézier curve optimisation")

        st.markdown("---")
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:#4af0c8;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.7rem;">📄 PDF Batch</div>""", unsafe_allow_html=True)
        max_pdf_pages   = st.slider("Max pages to scan", 1, 100, 50, 5)
        pdf_min_dim     = st.slider("Min image dimension (px)", 32, 200, 50, 10,
                                    help="Skip images smaller than this")
        bundle_linearts = st.checkbox("Bundle line arts in ZIP too", value=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:#7a7a99;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.8rem;">ε — Epsilon Guide</div>
        <div style="font-size:0.8rem;color:#e0e0f0;line-height:2;">
            <b style="color:#7cfc8e;">0.5</b> — Fine detail, large file<br>
            <b style="color:#7cfc8e;">1.0</b> — Balanced (default)<br>
            <b style="color:#7cfc8e;">3.0</b> — Clean geometric<br>
            <b style="color:#ffbb55;">5.0+</b> — Very coarse, tiny file
        </div>
        <hr style="border-color:#2a2a38;margin:1rem 0;">
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:#7a7a99;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:0.8rem;">Mode Reference</div>
        <div style="font-size:0.8rem;color:#e0e0f0;line-height:1.9;">
            <b style="color:#7cfc8e;">CANNY</b><br>
            Edge detection for photos and complex scenes.<br><br>
            <b style="color:#4af0c8;">ADAPTIVE</b><br>
            Best for documents, sketches, manuscripts.<br><br>
            <b style="color:#ffbb55;">XDOG</b><br>
            Extended DoG — stylized, painterly line art.
        </div>
        """, unsafe_allow_html=True)
else:
    # Sidebar closed — use defaults so downstream code doesn't crash
    turdsize = 2; alphamax = 1.0; opttol = 0.2
    max_pdf_pages = 50; pdf_min_dim = 50; bundle_linearts = True


def _toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open

# ─── Hero ─────────────────────────────────────────────────────────────────────
toggle_col, hero_col = st.columns([1, 9])
with toggle_col:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    arrow = "◀ HIDE" if st.session_state.sidebar_open else "▶▶ MENU"
    st.button(arrow, key="sidebar_toggle", use_container_width=True, on_click=_toggle_sidebar)
with hero_col:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">VectoLine</div>
        <div class="hero-subtitle">Image Vectorization Pipeline</div>
        <div class="pipeline-badge">
            Input Image <span>→</span> Line Art <span>→</span> SVG Vector <span>→</span> Simplified SVG
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Pipeline Settings — ALWAYS VISIBLE (no expander) ────────────────────────
st.markdown('<div class="pipeline-settings-wrap">', unsafe_allow_html=True)
st.markdown('<div class="pipeline-settings-title">⚙ Pipeline Settings</div>', unsafe_allow_html=True)

ps_col1, ps_col2, ps_col3 = st.columns([2, 2, 2])
with ps_col1:
    mode = st.selectbox("LINE ART MODE", options=["canny", "adaptive", "xdog"], index=0, key="mode_main")
with ps_col2:
    epsilon = st.slider("RDP EPSILON (simplification)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                        key="epsilon_main", help="Higher = fewer path points, smoother output")
with ps_col3:
    threshold = st.slider("THRESHOLD (xdog only)", min_value=64, max_value=200, value=128, step=8,
                          key="threshold_main")

st.markdown("""
<div class="mode-grid">
    <div class="mode-card"><strong>CANNY</strong>Edge detection. Great for photos.<br><span class="tag">photos</span></div>
    <div class="mode-card"><strong>ADAPTIVE</strong>Best for documents &amp; sketches.<br><span class="tag">documents</span></div>
    <div class="mode-card"><strong>XDOG</strong>Stylized, artistic line art.<br><span class="tag">artistic</span></div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

mode      = st.session_state.get("mode_main", "canny")
epsilon   = st.session_state.get("epsilon_main", 1.0)
threshold = st.session_state.get("threshold_main", 128)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_compare, tab_pdf = st.tabs([
    "  🖼  SINGLE IMAGE  ",
    "  🔬  MODE COMPARE  ",
    "  📄  PDF BATCH  ",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — Single Image
# ══════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a PNG, JPG, or BMP here",
                                type=["png","jpg","jpeg","bmp"],
                                key="single_upload", label_visibility="collapsed")

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w       = img_bgr.shape[:2]

        col_orig, col_la, col_svg = st.columns(3)
        with col_orig:
            st.markdown('<div class="img-panel-title">01 — ORIGINAL</div>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)
            st.caption(f"{w} × {h} px  ·  {uploaded.size // 1024} KB")

        run_btn = st.button("⚡  VECTORIZE", use_container_width=False)

        if run_btn:
            with st.spinner("Running pipeline…"):
                prog = st.progress(0, text="Step 1/3 — Line art…")
                line_art = image_to_line_art(img_bgr, mode=mode, threshold=threshold)
                prog.progress(33, text="Step 2/3 — Tracing with potrace…")
                with tempfile.TemporaryDirectory() as tmp:
                    raw_svg   = os.path.join(tmp, 'raw.svg')
                    final_svg = os.path.join(tmp, 'final.svg')
                    line_art_to_svg_adv(line_art, raw_svg, turdsize=turdsize, alphamax=alphamax, opttol=opttol)
                    size_kb    = os.path.getsize(raw_svg) / 1024
                    path_count = len(re.findall(r'<path', open(raw_svg).read()))
                    prog.progress(66, text="Step 3/3 — Simplifying paths (RDP)…")
                    content = open(raw_svg).read()
                    simplified, before, after = simplify_svg_content(content, epsilon)
                    open(final_svg, 'w').write(simplified)
                    svg_bytes = open(final_svg, 'rb').read()
                prog.progress(100, text="Done!")
                reduction = (1 - after / before) * 100 if before > 0 else 0

            with col_la:
                st.markdown('<div class="img-panel-title">02 — LINE ART</div>', unsafe_allow_html=True)
                st.image(line_art, use_container_width=True, clamp=True)
                st.caption(f"Mode: {mode.upper()}")

            with col_svg:
                st.markdown('<div class="img-panel-title">03 — VECTOR SVG</div>', unsafe_allow_html=True)
                svg_b64 = base64.b64encode(svg_bytes).decode()
                st.markdown(f'<img src="data:image/svg+xml;base64,{svg_b64}" style="width:100%;border-radius:8px;background:#fff;" />', unsafe_allow_html=True)
                st.caption(f"{size_kb:.1f} KB  ·  {path_count} paths")

            reduction_color = "#7cfc8e" if reduction > 0 else "#ffbb55"
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card"><div class="metric-val">{size_kb:.1f}<span style="font-size:0.9rem;color:#7a7a99"> KB</span></div><div class="metric-lbl">SVG Size</div></div>
                <div class="metric-card"><div class="metric-val">{path_count}</div><div class="metric-lbl">SVG Paths</div></div>
                <div class="metric-card"><div class="metric-val">{before:,}</div><div class="metric-lbl">Points Before</div></div>
                <div class="metric-card"><div class="metric-val">{after:,}</div><div class="metric-lbl">Points After</div></div>
                <div class="metric-card"><div class="metric-val" style="color:{reduction_color};">{reduction:.1f}<span style="font-size:0.9rem;color:#7a7a99">%</span></div><div class="metric-lbl">Reduction</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.download_button("⬇  Download SVG", data=svg_bytes,
                               file_name=f"{Path(uploaded.name).stem}_vector.svg",
                               mime="image/svg+xml")
    else:
        st.info("Upload an image above to get started.")


# ══════════════════════════════════════════════════════════
# TAB 2 — Mode Compare
# ══════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-label">Compare All 3 Line Art Modes Side-by-Side</div>', unsafe_allow_html=True)
    uploaded_cmp = st.file_uploader("Drop image here", type=["png","jpg","jpeg","bmp"],
                                    key="compare_upload", label_visibility="collapsed")

    if uploaded_cmp:
        file_bytes = np.frombuffer(uploaded_cmp.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        run_cmp = st.button("🔬  COMPARE ALL MODES")
        if run_cmp:
            with st.spinner("Generating all 3 modes…"):
                c_orig, c_canny, c_adapt, c_xdog = st.columns(4)
                labels = {
                    c_orig:  ("ORIGINAL",  None,       "#b8b8d0"),
                    c_canny: ("CANNY",     "canny",    "#7cfc8e"),
                    c_adapt: ("ADAPTIVE",  "adaptive", "#4af0c8"),
                    c_xdog:  ("XDOG",      "xdog",     "#ffbb55"),
                }
                for col, (title, m, color) in labels.items():
                    with col:
                        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:{color};letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">{title}</div>', unsafe_allow_html=True)
                        if m is None:
                            st.image(img_rgb, use_container_width=True)
                        else:
                            la = image_to_line_art(img_bgr, mode=m, threshold=threshold)
                            st.image(la, use_container_width=True, clamp=True)
    else:
        st.info("Upload an image to compare the three line art modes.")


# ══════════════════════════════════════════════════════════
# TAB 3 — PDF Batch
# ══════════════════════════════════════════════════════════
with tab_pdf:
    st.markdown('<div class="section-label">Extract & Vectorize All Images from a PDF</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Drop a PDF here", type=["pdf"],
                                    key="pdf_upload", label_visibility="collapsed")

    if uploaded_pdf:
        _, dl_col2 = st.columns([3, 1])
        with dl_col2:
            show_lineart = st.toggle("🎨 Show Line Art Preview", value=True)

        run_pdf = st.button("📄  PROCESS PDF")

        if run_pdf:
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, uploaded_pdf.name)
                open(pdf_path, 'wb').write(uploaded_pdf.read())
                doc = fitz.open(pdf_path)
                pages_to_scan = min(len(doc), max_pdf_pages)
                found = 0; svgs = {}; line_arts = {}; previews = []

                prog = st.progress(0, text=f"Scanning {pages_to_scan} pages…")
                for pn, page in enumerate(doc):
                    if pn >= pages_to_scan: break
                    for idx, info in enumerate(page.get_images(full=True)):
                        found += 1; label = f"p{pn+1}_img{idx+1}"
                        try:
                            bi  = doc.extract_image(info[0])
                            arr = np.frombuffer(bi['image'], np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is None: continue
                            h_i, w_i = img.shape[:2]
                            if min(h_i, w_i) < pdf_min_dim: continue

                            img_rgb_p = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            la = image_to_line_art(img, mode=mode, threshold=threshold)

                            if bundle_linearts:
                                la_buf = io.BytesIO(); Image.fromarray(la).save(la_buf, format='PNG')
                                line_arts[f"{label}_lineart.png"] = la_buf.getvalue()

                            raw_svg   = os.path.join(tmp, f'{label}_raw.svg')
                            final_svg = os.path.join(tmp, f'{label}.svg')
                            line_art_to_svg_adv(la, raw_svg, turdsize=turdsize, alphamax=alphamax, opttol=opttol)
                            content = open(raw_svg).read()
                            simp, _, _ = simplify_svg_content(content, epsilon)
                            open(final_svg, 'w').write(simp)
                            svg_bytes = open(final_svg, 'rb').read()
                            svgs[f"{label}.svg"] = svg_bytes
                            previews.append((label, img_rgb_p, la, base64.b64encode(svg_bytes).decode()))
                        except Exception as e:
                            st.warning(f"{label}: {e}")
                    prog.progress(int((pn+1)/pages_to_scan*100), text=f"Page {pn+1}/{pages_to_scan}")

                if svgs:
                    st.success(f"✅ Converted {len(svgs)} / {found} images")
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        svg_zip = io.BytesIO()
                        with zipfile.ZipFile(svg_zip, 'w') as zf:
                            for name, data in svgs.items(): zf.writestr(name, data)
                        svg_zip.seek(0)
                        st.download_button(f"⬇  Download {len(svgs)} SVGs as ZIP",
                                           data=svg_zip.getvalue(),
                                           file_name=f"{Path(uploaded_pdf.name).stem}_svgs.zip",
                                           mime="application/zip", use_container_width=True)
                    if bundle_linearts and line_arts:
                        with btn_col2:
                            la_zip = io.BytesIO()
                            with zipfile.ZipFile(la_zip, 'w') as zf:
                                for name, data in line_arts.items(): zf.writestr(name, data)
                            la_zip.seek(0)
                            st.download_button(f"⬇  Download {len(line_arts)} Line Arts as ZIP",
                                               data=la_zip.getvalue(),
                                               file_name=f"{Path(uploaded_pdf.name).stem}_linearts.zip",
                                               mime="application/zip", use_container_width=True)

                    if show_lineart and previews:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Line Art Preview</div>', unsafe_allow_html=True)
                        st.markdown("""
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:0.5rem;text-align:center;">
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#7a7a99;letter-spacing:2px;">ORIGINAL</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#7cfc8e;letter-spacing:2px;">LINE ART</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#4af0c8;letter-spacing:2px;">VECTOR SVG</div>
                        </div>""", unsafe_allow_html=True)
                        for label, orig_rgb, la_arr, svg_b64 in previews:
                            c1, c2, c3 = st.columns(3)
                            with c1: st.image(orig_rgb, use_container_width=True, caption=label)
                            with c2: st.image(la_arr, use_container_width=True, clamp=True, caption=f"{mode.upper()} line art")
                            with c3:
                                st.markdown(f'<img src="data:image/svg+xml;base64,{svg_b64}" style="width:100%;border-radius:6px;background:#fff;" />', unsafe_allow_html=True)
                                st.caption("Vector SVG")
                            st.markdown("<hr style='border-color:#2a2a38;margin:0.5rem 0;'>", unsafe_allow_html=True)
                else:
                    st.warning("No images found or extractable in this PDF.")
    else:
        st.info("Upload a PDF to batch-vectorize all embedded images.")
