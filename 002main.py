import datetime
import os
import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="Pancreas Liquid CGP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #667085;
        --primary: #6558e8;
        --secondary: #16a6a1;
        --surface: rgba(255, 255, 255, .82);
        --line: rgba(101, 88, 232, .13);
    }

    html {
        color-scheme: light !important;
        background: #f7f8fc;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    body { background: #f7f8fc; }
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(101, 88, 232, .13), transparent 29rem),
            radial-gradient(circle at 92% 18%, rgba(22, 166, 161, .12), transparent 28rem),
            #f7f8fc;
        color: var(--ink);
    }
    .stApp,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    .stApp input {
        color: var(--ink) !important;
        background-color: white !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    .block-container { max-width: 1180px; padding: 2.4rem 2rem 4rem; }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.3rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,.72);
        border-radius: 28px;
        color: white;
        background: linear-gradient(130deg, #292351 0%, #5146bd 55%, #148f91 120%);
        box-shadow: 0 24px 60px rgba(47, 39, 105, .20);
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 18rem; height: 18rem; right: -5rem; top: -8rem;
        border-radius: 50%; border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 0 0 3rem rgba(255,255,255,.035), 0 0 0 6rem rgba(255,255,255,.025);
    }
    .eyebrow {
        display: inline-block; padding: .38rem .72rem; margin-bottom: .85rem;
        border: 1px solid rgba(255,255,255,.24); border-radius: 999px;
        background: rgba(255,255,255,.10); font-size: .74rem; font-weight: 700;
        letter-spacing: .09em; text-transform: uppercase;
    }
    .hero, .hero h1, .hero .eyebrow { color: white !important; }
    .hero h1 { margin: 0; max-width: 760px; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.05; letter-spacing: -.045em; }
    .hero p { max-width: 700px; margin: 1rem 0 0; color: rgba(255,255,255,.78); font-size: 1rem; line-height: 1.65; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface); border-color: var(--line) !important;
        border-radius: 20px; box-shadow: 0 12px 34px rgba(33, 43, 72, .055);
    }
    h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    div[data-testid="stNumberInput"] input, div[data-testid="stDateInput"] input {
        border-radius: 10px; background: rgba(255,255,255,.9);
    }
    div[role="radiogroup"] { gap: .35rem; }
    div[role="radiogroup"] label {
        padding: .3rem .7rem; border: 1px solid #e6e8f0; border-radius: 10px;
        color: var(--ink) !important; background: white;
    }
    .stButton > button {
        width: 100%; min-height: 3.25rem; border: 0; border-radius: 14px;
        color: white; font-weight: 700; font-size: 1rem;
        background: linear-gradient(100deg, var(--primary), #7b6ff0 55%, var(--secondary));
        box-shadow: 0 10px 25px rgba(101,88,232,.25); transition: .2s ease;
    }
    .stButton > button:hover { color: white; transform: translateY(-2px); box-shadow: 0 14px 30px rgba(101,88,232,.30); }
    .section-kicker { margin: -.55rem 0 .9rem; color: var(--muted); font-size: .86rem; }
    .result-card {
        padding: 1.4rem 1.5rem; border: 1px solid rgba(101,88,232,.15); border-radius: 18px;
        background: linear-gradient(135deg, rgba(101,88,232,.08), rgba(22,166,161,.06));
    }
    .result-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .result-value { margin: .25rem 0; font-size: 3.5rem; line-height: 1; font-weight: 800; letter-spacing: -.06em; color: var(--primary); }
    .result-copy { color: var(--muted); font-size: .9rem; }
    .disclaimer {
        margin-top: 1rem; padding: .85rem 1rem; border-left: 3px solid #f0a33a;
        border-radius: 6px 12px 12px 6px; color: #6d542d; background: #fff8e9; font-size: .84rem;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 3rem; }
        .hero { padding: 1.7rem 1.4rem; border-radius: 20px; }
        .hero h1 { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <span class="eyebrow">Clinical Prediction Tool</span>
        <h1>Pancreas Liquid CGP</h1>
        <p>Estimate the probability of ctDNA detection by liquid comprehensive genomic profiling in pancreatic adenocarcinoma.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Patient profile")
        st.markdown('<p class="section-kicker">Basic demographic and performance information</p>', unsafe_allow_html=True)
        basic_1, basic_2 = st.columns(2, gap="medium")
        with basic_1:
            age = st.number_input("Age", min_value=0, max_value=100, value=0, step=1)
            sex = st.radio("Sex", ["Woman", "Man"], horizontal=True)
        with basic_2:
            ps = st.radio("ECOG performance status", ["0", "1", "2", "3", "4"], horizontal=True)

    with st.container(border=True):
        st.subheader("Clinical timeline")
        st.markdown('<p class="section-kicker">Dates and latest treatment response</p>', unsafe_allow_html=True)
        date_1, date_2 = st.columns(2, gap="medium")
        with date_1:
            diagdate = st.date_input("Diagnosis date", datetime.date.today())
        with date_2:
            spedate = st.date_input("Specimen collection date", datetime.date.today())
        treatmentline = st.number_input("Current treatment line", min_value=0, max_value=20, value=0, step=1)
        response = st.radio("Response", ["PD", "SD", "PR", "CR", "NE"], horizontal=True)

with right:
    with st.container(border=True):
        st.subheader("Metastatic sites")
        st.markdown('<p class="section-kicker">Select all sites that apply</p>', unsafe_allow_html=True)
        meta_1, meta_2 = st.columns(2)
        with meta_1:
            lymphmeta = st.toggle("Lymph node", value=False)
            lungmeta = st.toggle("Lung", value=False)
            pleuralmeta = st.toggle("Pleura", value=False)
            livermeta = st.toggle("Liver", value=False)
            bonemeta = st.toggle("Bone", value=False)
            brainmeta = st.toggle("Brain", value=False)
        with meta_2:
            peritonealmeta = st.toggle("Peritoneum", value=False)
            kidneymeta = st.toggle("Kidney", value=False)
            adrenalsmeta = st.toggle("Adrenal", value=False)
            musclemeta = st.toggle("Muscle", value=False)
            softmeta = st.toggle("Soft tissue", value=False)
            ovarymeta = st.toggle("Ovary", value=False)

    button = st.button("Calculate detection probability", type="primary")

if button:
    sexinput = 0 if sex == "Woman" else 1

    ps0 = 1 if ps == "0" else 0
    ps1 = 1 if ps == "1" else 0
    ps2 = 1 if ps == "2" else 0
    ps3 = 1 if ps == "3" else 0
    ps4 = 1 if ps == "4" else 0

    spediag = (spedate - diagdate).days
    if spediag < 0:
        spediag = 0

    pd = 1 if response == "PD" else 0
    sd = 1 if response == "SD" else 0
    pr = 1 if response == "PR" else 0
    cr = 1 if response == "CR" else 0
    ne = 1 if response == "NE" else 0

    metasite = sum(
        int(site)
        for site in [
            lymphmeta, lungmeta, pleuralmeta, livermeta, bonemeta, brainmeta,
            peritonealmeta, kidneymeta, adrenalsmeta, musclemeta, softmeta, ovarymeta,
        ]
    )

    model_input = [
        sexinput, age, spediag, metasite,
        int(lymphmeta), int(lungmeta), int(pleuralmeta), int(livermeta), int(bonemeta), int(brainmeta),
        int(peritonealmeta), int(kidneymeta), int(adrenalsmeta), int(musclemeta), int(softmeta), int(ovarymeta),
        treatmentline, ps0, ps1, ps2, ps3, ps4, cr, ne, pd, pr, sd,
    ]

    pred = 0
    fold = 5
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pancreasliquidmodel")
    with st.spinner("Running the prediction models…"):
        for i in range(fold):
            model_path = os.path.join(model_dir, f"logistic_250401_shap_{i}.pkl")
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
            pred += model.predict_proba([model_input])[:, 1][0] / fold
    pred *= 100

    if pred < 0.01:
        pred = 0.01
    elif pred > 99.99:
        pred = 99.99

    st.markdown("### Prediction result")
    result_1, result_2 = st.columns([1, 1.25], gap="large")
    with result_1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">ctDNA detection probability</div>
                <div class="result-value">{pred:.1f}<span style="font-size:1.7rem">%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sizes = [pred, 100 - pred]
    fig, ax = plt.subplots(figsize=(5.2, 2.4), facecolor="none")
    ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        colors=["#6558e8", "#e7e9f2"],
        wedgeprops={"width": 0.22, "edgecolor": "none"},
    )
    ax.text(0, 0.08, f"{pred:.1f}%", ha="center", va="center", fontsize=23, fontweight="bold", color="#172033")
    ax.text(0, -0.22, "probability", ha="center", va="center", fontsize=9, color="#667085")
    ax.axis("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=160)
    plt.close(fig)
    with result_2:
        st.image(buf, width="stretch")

    st.markdown(
        '<div class="disclaimer"><strong>Clinical notice</strong><br>This result cannot be used for clinical diagnosis. Please consider performing CGP tests at a physician\'s discretion.</div>',
        unsafe_allow_html=True,
    )
