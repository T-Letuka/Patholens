

import sys
import random
import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go


sys.path.append(str(Path(__file__).parent.parent))
from model import load_model_for_inference, get_device, CLASS_NAMES
from report_generator import predict_and_report, CLASS_CLINICAL_INFO

#icon 
#streamlit doesnt behave like normal html, so i am stuck with these emojis from windows.

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="PathoLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0d1321;
    --bg2:        #0F1623;
    --bg3:        #161E30;
    --bg4:        #1C2438;
    --light_purple   :#cdb4db;
    --red:        #FF4B6E;
    --amber:      #FFB547;
    --blue:       #4B9EFF;
    --purple:     #A78BFA;
    --t1:         #EEF2FF;
    --t2:         #94A3B8;
    --t3:         #64748B;
    --border:     #1C2438;
    --border2:    #2A3650;
}

.stApp { background: var(--bg) !important; font-family: 'Outfit', sans-serif; }
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border2);
}

/* Typography */
h1,h2,h3 { font-family: 'DM Serif Display', serif !important; color: var(--t1) !important; }
p, li { color: var(--t2); }

/* ── HERO ──────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 60%, #080D18 100%);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60%; right: -5%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(0,212,170,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem; color: #EEF2FF; margin: 0; line-height: 1.1;
}
.hero-title em { color: var(--light_purple); font-style: normal; }
.hero-sub {
    font-size: 0.9rem; color: var(--t3); margin-top: 0.4rem;
    font-weight: 300; letter-spacing: 0.02em;
}
.badges { display: flex; gap: 0.4rem; margin-top: 1rem; flex-wrap: wrap; }
.badge {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    padding: 0.2rem 0.6rem; border-radius: 999px; border: 1px solid;
    font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
}
.bt { color: var(--light_purple);   border-color: rgba(0,212,170,0.3);  background: rgba(0,212,170,0.06); }
.bb { color: var(--blue);   border-color: rgba(75,158,255,0.3); background: rgba(75,158,255,0.06); }
.ba { color: var(--amber);  border-color: rgba(255,181,71,0.3); background: rgba(255,181,71,0.06); }
.bp { color: var(--purple); border-color: rgba(167,139,250,0.3);background: rgba(167,139,250,0.06); }

/* ── PANELS ────────────────────────────── */
.panel {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 1.5rem;
    height: 80%;
}
.panel-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: var(--light_purple); letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.panel-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem; color: var(--t1); margin-bottom: 1rem;
}

/* ── SPECIMEN CARD ─────────────────────── */
.specimen-meta {
    background: var(--bg4);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-top: 0.8rem;
}
.meta-row {
    display: flex; justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}
.meta-row:last-child { border-bottom: none; }
.meta-key   { color: var(--t3); font-family: 'DM Mono', monospace; }
.meta-val   { color: var(--t2); font-weight: 500; }
.specimen-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: var(--light_purple);
    background: rgba(0,212,170,0.06);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 4px; padding: 0.15rem 0.5rem;
    display: inline-block; margin-bottom: 0.6rem;
}

/* ── UPLOAD ZONE ───────────────────────── */
.upload-success {
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    display: flex; align-items: center; gap: 0.6rem;
    margin-top: 0.8rem;
}
.upload-success-icon { font-size: 1.1rem; }
.upload-success-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: var(--light_purple);
}

/* ── RESULTS — BANNER ──────────────────── */
.result-banner {
    border-radius: 12px; padding: 1.5rem 2rem;
    border: 1px solid; margin-bottom: 1.5rem;
}
.result-banner.mal  { background: rgba(255,75,110,0.07);  border-color: rgba(255,75,110,0.35); }
.result-banner.ben  { background: rgba(0,212,170,0.07);   border-color: rgba(0,212,170,0.35); }
.result-banner.ind  { background: rgba(255,181,71,0.07);  border-color: rgba(255,181,71,0.35); }
.result-tag {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.result-class {
    font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0;
}
.result-conf {
    font-family: 'DM Mono', monospace; font-size: 0.82rem;
    color: var(--t3); margin-top: 0.25rem;
}

/* ── METRIC TILES ──────────────────────── */
.tile {
    background: var(--bg4); border: 1px solid var(--border2);
    border-radius: 8px; padding: 0.9rem; text-align: center;
}
.tile-val {
    font-family: 'DM Mono', monospace; font-size: 1.5rem;
    font-weight: 500; color: var(--light_purple);
}
.tile-lbl {
    font-size: 0.7rem; color: var(--t3);
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.15rem;
}

/* ── REPORT ────────────────────────────── */
.report-wrap {
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 12px; padding: 2rem;
    font-family: 'Outfit', sans-serif;
}
.report-wrap h2,.report-wrap h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--t1) !important;
}
.report-wrap p,.report-wrap li { color: var(--t2); line-height: 1.7; }
.report-wrap table { width: 100%; border-collapse: collapse; }
.report-wrap td,.report-wrap th {
    padding: 0.45rem 0.7rem;
    border: 1px solid var(--border2); color: var(--t2); font-size: 0.85rem;
}
.report-wrap blockquote {
    border-left: 3px solid var(--amber);
    padding-left: 1rem; color: var(--amber) !important; font-size: 0.82rem;
}

/* ── BUTTONS ───────────────────────────── */
.stButton > button {
    background: var(--bg4) !important; color: var(--t1) !important;
    border: 1px solid var(--border2) !important; border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    border-color: var(--light_purple) !important; color: var(--light_purple) !important;
    background: rgba(0,212,170,0.06) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg4) !important;
    border: 1px dashed var(--border2) !important; border-radius: 10px !important;
}
hr { border-color: var(--border2) !important; margin: 1.2rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_DIR = Path(__file__).parent / "sample_images"
MODEL_PATH = Path(__file__).parent.parent / "models" / "patholens_best_model.pth"

CLASS_LABELS = {
    "Colon Adenocarcinoma":         {"short": "Colon ACA",    "malignant": True,  "site": "Colon", "icd": "C18.9"},
    "Colon Benign":                 {"short": "Colon Benign", "malignant": False, "site": "Colon", "icd": "K63.9"},
    "Lung Adenocarcinoma":          {"short": "Lung ACA",     "malignant": True,  "site": "Lung",  "icd": "C34.1"},
    "Lung Benign":                  {"short": "Lung Benign",  "malignant": False, "site": "Lung",  "icd": "J98.4"},
    "Lung Squamous Cell Carcinoma": {"short": "Lung SCC",     "malignant": True,  "site": "Lung",  "icd": "C34.1"},
}

FOLDER_TO_CLASS = {
    "colon_aca": "Colon Adenocarcinoma",
    "colon_n":   "Colon Benign",
    "lung_aca":  "Lung Adenocarcinoma",
    "lung_n":    "Lung Benign",
    "lung_scc":  "Lung Squamous Cell Carcinoma",
}

# Maps folder prefix to anonymous specimen ID shown to user
# Hides the class identity until after classification
SPECIMEN_IDS = {
    "colon_aca": "SPEC-COL-A",
    "colon_n":   "SPEC-COL-B",
    "lung_aca":  "SPEC-LNG-A",
    "lung_n":    "SPEC-LNG-B",
    "lung_scc":  "SPEC-LNG-C",
}

TISSUE_SITES = {
    "colon_aca": "Colon",
    "colon_n":   "Colon",
    "lung_aca":  "Lung",
    "lung_n":    "Lung",
    "lung_scc":  "Lung",
}


# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource
def load_model():
    device = get_device()
    if not MODEL_PATH.exists():
        return None, None
    model = load_model_for_inference(MODEL_PATH, device)
    return model, device


@st.cache_data
def get_sample_images():
    if not SAMPLE_DIR.exists():
        return []
    samples = []
    for img_path in sorted(SAMPLE_DIR.glob("*.jpeg")) + sorted(SAMPLE_DIR.glob("*.jpg")):
        prefix     = img_path.stem.split("__")[0]
        class_name = FOLDER_TO_CLASS.get(prefix, "Unknown")
        site       = TISSUE_SITES.get(prefix, "Unknown")
        spec_id    = SPECIMEN_IDS.get(prefix, "SPEC-UNK")
        samples.append({
            "path":       img_path,
            "class_name": class_name,
            "site":       site,
            "spec_id":    spec_id,
            "prefix":     prefix,
        })
    return samples


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(image: Image.Image, model, device) -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


# =============================================================================
# PROBABILITY CHART
# =============================================================================

def make_prob_chart(probabilities: dict) -> go.Figure:
    classes = list(probabilities.keys())
    probs   = [probabilities[c] * 100 for c in classes]
    short   = [CLASS_LABELS[c]["short"] for c in classes]
    colors  = ["#ff0000" if CLASS_LABELS[c]["malignant"] else "#a1ff0a" for c in classes]

    fig = go.Figure(go.Bar(
        x=probs, y=short, orientation='h',
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(family="DM Mono", size=11, color="#64748B"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor="rgba(42,54,80,0.6)",
                   tickfont=dict(color="#64748B", family="DM Mono", size=10),
                   ticksuffix="%", zeroline=False),
        yaxis=dict(tickfont=dict(color="#94A3B8", family="Outfit", size=11),
                   autorange="reversed"),
        margin=dict(l=5, r=65, t=5, b=5), height=210, showlegend=False,
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:0.8rem 0'>
            <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#EEF2FF;'>
                🔬 PathoLens
            </div>
            <div style='font-size:0.7rem;color:#64748B;margin-top:0.2rem;
                        font-family:DM Mono,monospace;'>
                v1.0 · EfficientNet-B0
            </div>
        </div>
        <hr style='border-color:#1C2438;margin:0.5rem 0 1rem;'>
        """, unsafe_allow_html=True)

        st.markdown("**Model Performance**")
        for val, lbl in [("100%","Test Accuracy"),("1.000","Macro AUC-ROC"),
                          ("100%","Sensitivity"),("100%","Specificity")]:
            st.markdown(f"""
            <div class='tile' style='margin-bottom:0.45rem;'>
                <div class='tile-val'>{val}</div>
                <div class='tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Tissue Classes**")
        for cls, info in CLASS_LABELS.items():
            colour = "#FF4B6E" if info["malignant"] else "#00D4AA"
            label  = "Malignant" if info["malignant"] else "Benign"
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.35rem 0;border-bottom:1px solid #1C2438;'>
                <span style='font-size:0.8rem;color:#94A3B8;'>{info['short']}</span>
                <span style='font-family:DM Mono,monospace;font-size:0.65rem;
                             color:{colour};background:rgba(0,0,0,0.2);
                             border:1px solid {colour}44;border-radius:3px;
                             padding:0.1rem 0.4rem;'>{label}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Dataset**")
        st.markdown("""
        <div style='font-size:0.78rem;color:#64748B;line-height:1.65;'>
        LC25000 · 25,000 H&E images<br>
        5 tissue classes · 768×768px<br>
        EfficientNet-B0 fine-tuned<br>
        PyTorch · timm · Anthropic API
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem;color:#64748B;line-height:1.5;
                    padding:0.7rem;background:rgba(255,181,71,0.05);
                    border:1px solid rgba(255,181,71,0.2);border-radius:6px;'>
        ⚠ Research prototype only.<br>
        Not approved for clinical use.<br>
        Requires pathologist review.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# HERO
# =============================================================================

def render_hero():
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>Patho<em>Lens</em></div>
        <div class='hero-sub'>
            AI-Assisted Histopathology Classification ·
            EfficientNet-B0 · LC25000 · H&E Stained Tissue
        </div>
        <div class='badges'>
            <span class='badge bt'>Deep Learning</span>
            <span class='badge bb'>Transfer Learning</span>
            <span class='badge bt'>Computer Vision</span>
            <span class='badge ba'>Clinical NLP</span>
            <span class='badge bb'>100% Test Accuracy</span>
            <span class='badge bp'>AUC-ROC 1.000</span>
        </div>
    </div>""", unsafe_allow_html=True)


# =============================================================================
# VIEW A — UPLOAD VIEW
# =============================================================================

def render_upload_view():
    samples = get_sample_images()
    if not samples:
        st.error("Sample images not found at app/sample_images/")
        return

  
    if "sample" not in st.session_state:
        st.session_state.sample = random.choice(samples)
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    col_left, col_right = st.columns([1, 1], gap="large")

  
    with col_left:
        st.markdown("""
        <div class='panel-label'>Step 01</div>
        <div class='panel-title' style='font-family:DM Serif Display,serif;
             font-size:1.25rem;color:#EEF2FF;margin-bottom:1rem;'>
            Get a Sample Slide
        </div>""", unsafe_allow_html=True)

        if st.button("🎲  Randomise Sample", use_container_width=True):
            
            other = [s for s in samples if s["path"] != st.session_state.sample["path"]]
            st.session_state.sample = random.choice(other if other else samples)

        sample = st.session_state.sample
        img    = Image.open(sample["path"])

        st.image(img, use_container_width=True)

        
        st.markdown(f"""
        <div class='specimen-meta'>
            <div class='specimen-id'>{sample['spec_id']}</div>
            <div class='meta-row'>
                <span class='meta-key'>Tissue Site</span>
                <span class='meta-val'>{sample['site']}</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Stain</span>
                <span class='meta-val'>Haematoxylin & Eosin (H&E)</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Resolution</span>
                <span class='meta-val'>768 × 768 px</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Source</span>
                <span class='meta-val'>LC25000 Test Set</span>
            </div>
        </div>""", unsafe_allow_html=True)

       
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)

        st.download_button(
            label="⬇  Download Slide",
            data=img_bytes,
            file_name=f"{sample['spec_id']}.jpeg",
            mime="image/jpeg",
            use_container_width=True,
        )

        st.markdown("""
        <div style='font-size:0.72rem;color:#64748B;margin-top:0.4rem;
                    text-align:center;font-style:italic;'>
            Download the slide, then upload it for classification →
        </div>""", unsafe_allow_html=True)

    
    with col_right:
        st.markdown("""
        <div class='panel-label'>Step 02</div>
        <div class='panel-title' style='font-family:DM Serif Display,serif;
             font-size:1.25rem;color:#EEF2FF;margin-bottom:1rem;'>
            Upload for Classification
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload the downloaded slide — or any H&E histology image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        # Upload success notifier
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded
            file_size_kb = round(uploaded.size / 1024, 1)
            st.markdown(f"""
            <div class='upload-success'>
                <span class='upload-success-icon'>✅</span>
                <div>
                    <div class='upload-success-text'>
                        Image received — ready for analysis
                    </div>
                    <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                                color:#64748B;margin-top:0.15rem;'>
                        {uploaded.name} · {file_size_kb} KB · JPEG
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Preview thumbnail
            preview = Image.open(uploaded).convert("RGB")
            st.image(preview, caption="Uploaded image preview", use_container_width=True)

        else:
            # Placeholder when nothing uploaded yet
            st.markdown("""
            <div style='border:1px dashed #2A3650;border-radius:10px;
                        padding:3rem 1rem;text-align:center;margin-top:0.5rem;'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>🔬</div>
                <div style='font-family:DM Mono,monospace;font-size:0.75rem;
                            color:#64748B;'>
                    Waiting for upload...
                </div>
                <div style='font-size:0.72rem;color:#374151;margin-top:0.3rem;'>
                    Accepts JPG / PNG
                </div>
            </div>""", unsafe_allow_html=True)

        
        st.markdown("<br>", unsafe_allow_html=True)

        analyse_disabled = uploaded is None

        if not analyse_disabled:
            if st.button("🧬  Analyse Tissue →", use_container_width=True, type="primary"):
                # Store everything needed for results view
                st.session_state.view            = "results"
                st.session_state.uploaded_file   = uploaded
                st.session_state.report_ready    = False
                st.rerun()
        else:
            st.markdown("""
            <div style='background:#1C2438;border:1px solid #2A3650;
                        border-radius:8px;padding:0.75rem;text-align:center;
                        color:#374151;font-size:0.85rem;'>
                Upload an image to enable analysis
            </div>""", unsafe_allow_html=True)


# =============================================================================
# VIEW B — RESULTS VIEW
# =============================================================================

def render_results_view(model, device):
    uploaded = st.session_state.get("uploaded_file")

    if uploaded is None:
        st.session_state.view = "upload"
        st.rerun()
        return

    # Back button
    if st.button("← Analyse Another", use_container_width=False):
        st.session_state.view         = "upload"
        st.session_state.report_ready = False
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Load image and run inference
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Classifying tissue..."):
        probabilities = run_inference(image, model, device)

    predicted_class = max(probabilities, key=probabilities.get)
    confidence      = probabilities[predicted_class]
    info            = CLASS_LABELS[predicted_class]
    is_malignant    = info["malignant"]
    is_low_conf     = confidence < 0.80

    # ── RESULT BANNER ──────────────────────────────────────────────────────────
    if is_low_conf:
        b_class = "ind"; b_color = "#FFB547"
        b_tag   = "⚠  INDETERMINATE — EXPERT REVIEW REQUIRED"
    elif is_malignant:
        b_class = "mal"; b_color = "#d00000"
        b_tag   = "🔴  MALIGNANT TISSUE IDENTIFIED"
    else:
        b_class = "ben"; b_color = "#548c2f"
        b_tag   = "🟢  NO EVIDENCE OF MALIGNANCY"

    st.markdown(f"""
    <div class='result-banner {b_class}'>
        <div class='result-tag' style='color:{b_color};'>{b_tag}</div>
        <div class='result-class' style='color:{b_color};'>{predicted_class}</div>
        <div class='result-conf'>
            Confidence: {confidence*100:.1f}%
            {"&nbsp;·&nbsp;⚠ Below 80% threshold — differential diagnosis below" if is_low_conf else ""}
        
    """, unsafe_allow_html=True)

    # ── TWO COLUMN LAYOUT ──────────────────────────────────────────────────────
    col_img, col_chart = st.columns([1, 1.3], gap="large")

    with col_img:
        st.image(image, use_container_width=True, caption="Analysed slide")

        # Classification metadata
        icd   = info["icd"]
        site  = info["site"]
        ctype = "Malignant" if is_malignant else "Benign"

        st.markdown(f"""
        <div class='specimen-meta' style='margin-top:0.8rem;'>
            <div class='meta-row'>
                <span class='meta-key'>Classification</span>
                <span class='meta-val' style='color:{b_color};font-weight:600;'>{ctype}</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Tissue Site</span>
                <span class='meta-val'>{site}</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>ICD-10 Code</span>
                <span class='meta-val' style='font-family:DM Mono,monospace;'>{icd}</span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Confidence</span>
                <span class='meta-val' style='font-family:DM Mono,monospace;'>
                    {confidence*100:.1f}%
                </span>
            </div>
            <div class='meta-row'>
                <span class='meta-key'>Stain</span>
                <span class='meta-val'>H&E</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown("""
        <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                    color:#64748B;letter-spacing:0.1em;text-transform:uppercase;
                    margin-bottom:0.5rem;'>
            Class Probabilities
        </div>""", unsafe_allow_html=True)

        fig = make_prob_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Model info tiles
        t1, t2, t3 = st.columns(3)
        for col, val, lbl in [
            (t1, f"{confidence*100:.1f}%", "Confidence"),
            (t2, info["icd"],              "ICD-10"),
            (t3, "5-Class",                "Model Output"),
        ]:
            col.markdown(f"""
            <div class='tile'>
                <div class='tile-val' style='font-size:1.1rem;'>{val}</div>
                <div class='tile-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── CLINICAL REPORT ────────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                color:#00D4AA;letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:0.8rem;'>
                   
       📋  Clinical Report Generation
    </div>""", unsafe_allow_html=True)

    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY", None)
        or __import__("os").environ.get("ANTHROPIC_API_KEY", None)
    )

    if not api_key:
        st.warning(
            "Anthropic API key not configured. "
            "Add ANTHROPIC_API_KEY to .streamlit/secrets.toml"
        )
        return

    if not st.session_state.get("report_ready"):
        if st.button("📄  Generate Clinical Report", use_container_width=True):
            with st.spinner("Generating preliminary pathology report..."):
                try:
                    report_dict, report_md = predict_and_report(
                        class_probabilities=probabilities,
                        image_filename=uploaded.name,
                        api_key=api_key,
                    )
                    st.session_state.report_md    = report_md
                    st.session_state.report_dict  = report_dict
                    st.session_state.report_ready = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

    if st.session_state.get("report_ready"):
        st.markdown(
            f"<div class='report-wrap'>{st.session_state.report_md}</div>",
            unsafe_allow_html=True
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="⬇  Download Report (.md)",
                data=st.session_state.report_md,
                file_name=f"patholens_report_{predicted_class.replace(' ','_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            if st.button("← Analyse Another Slide", use_container_width=True):
                st.session_state.view         = "upload"
                st.session_state.report_ready = False
                st.rerun()

## happened to watch a very old video using streamlit and now , there retired stuff , i dont know how its still running btw
# =============================================================================
# MAIN
# =============================================================================

def main():
    render_sidebar()
    render_hero()

    model, device = load_model()
    if model is None:
        st.error("Model not found at models/best_model.pth")
        return

    # Initialise view state
    if "view" not in st.session_state:
        st.session_state.view = "upload"

    if st.session_state.view == "upload":
        render_upload_view()
    else:
        render_results_view(model, device)

    # Footer
    st.markdown("""
    <div style='text-align:center;padding:2.5rem 0 0.5rem;
                font-family:DM Mono,monospace;font-size:0.68rem;color:#374151;'>
        PathoLens · EfficientNet-B0 · LC25000 ·
        Tisetso Letuka ·
        BSc Biomedical Science · Honours Anatomical Pathology 
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
