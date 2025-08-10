# streamlit_app.py
# Dark theme (bg #000000, text #FFFFFF)
# Loads your local model automatically and predicts (no sidebar/settings).

import os, io, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Stroke Risk • drfin.", page_icon="💙", layout="centered")

# ---------- DARK CSS ----------
CSS = """
:root{
  --bg:#000000; --card:#0B0B0B; --text:#FFFFFF; --muted:#B0B8C5;
  --border:#1F2937; --input:#111317; --primary:#3B82F6; --accent:#06B6D4;
  --radius:14px; --shadow:0 8px 24px rgba(0,0,0,.55); --maxw:1080px;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }

/* container */
.container{ max-width:var(--maxw); margin:0 auto; padding:1rem 1.25rem 2rem; }

/* hero */
.hero{ text-align:center; margin:.6rem 0 1.1rem; }
.hero .brand{ font-weight:800; color:var(--text); margin-bottom:.3rem; }
.hero .icon{ font-size:1.6rem; }
.hero .title{
  font-size:2rem; font-weight:800; line-height:1.15;
  background:linear-gradient(90deg,var(--accent),var(--primary));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin:.2rem 0 .4rem;
}
.hero .sub{ color:var(--muted); font-size:.98rem; }

/* card */
.card{
  background:var(--card); border:1px solid var(--border);
  border-radius:var(--radius); box-shadow:var(--shadow); padding:1.1rem;
}
.section{ display:flex; align-items:center; gap:.5rem; margin-bottom:.8rem; }
.section .title{ font-weight:800; color:var(--text); font-size:1.04rem; }

/* labels */
.stTextInput label, .stNumberInput label, .stSelectbox label {
  color:var(--text) !important; font-weight:600;
}

/* inputs */
.stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div{
  background:var(--input); border:1px solid var(--border); border-radius:12px;
}
[data-baseweb="select"] > div{
  background:var(--input); border:1px solid var(--border); border-radius:12px; color:var(--text);
}
input[type="text"], input[type="number"], textarea{ color:var(--text); }
::placeholder{ color:#9CA3AF; }

/* action button */
.stButton > button{
  width:100%; border:none; border-radius:999px; padding:.75rem 1rem; font-weight:800; color:#fff;
  background:linear-gradient(90deg,var(--accent),var(--primary));
  box-shadow:0 12px 28px rgba(59,130,246,.28);
}

/* metrics */
.metrics{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
@media (max-width: 780px){ .metrics{ grid-template-columns:1fr; } }
.metric{
  background:var(--card); border:1px solid var(--border);
  border-radius:var(--radius); padding:1rem; text-align:center; box-shadow:var(--shadow);
}
.metric .t{ color:var(--muted); font-size:.92rem; margin-bottom:.3rem; }
.metric .v{ color:var(--text); font-size:1.75rem; font-weight:800; }

/* gauge */
.gauge{ --val:0.0; width:132px; height:132px; margin:0 auto; position:relative;
  background:radial-gradient(closest-side,var(--card) 79%,transparent 80% 100%),
             conic-gradient(var(--primary) calc(var(--val)*360deg), #2A2F37 0);
  border-radius:50%; box-shadow:var(--shadow);
}
.gauge .c{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-weight:800; color:var(--text); }

/* chip */
.chip{ display:inline-block; margin-top:.6rem; padding:.45rem .75rem; border-radius:999px;
  font-weight:700; border:1px solid var(--border); background:var(--card); color:var(--text); }
.chip.ok{ color:#22C55E; border-color:rgba(34,197,94,.35); background:rgba(34,197,94,.08); }
.chip.bad{ color:#F87171; border-color:rgba(248,113,113,.35); background:rgba(248,113,113,.08); }

/* small note */
.note{ color:var(--muted); font-size:.85rem; margin-top:.35rem; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------- hero ----------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <div class="brand">drfin.</div>
      <div class="title">Stroke Risk Prediction System</div>
      <div class="sub">Advanced AI-powered stroke risk assessment tool for healthcare professionals.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- model helpers ----------
EXPECTED_COLS = ["gender","age","hypertension","heart_disease","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]
DEFAULT_MODEL_PATHS = [
    "logreg_stroke_pipeline.joblib",
    "/mnt/data/logreg_stroke_pipeline.joblib",
]

def find_model_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_model_auto():
    path = find_model_path(DEFAULT_MODEL_PATHS)
    if path is None:
        raise FileNotFoundError(
            "Model file not found. Put 'logreg_stroke_pipeline.joblib' in the app folder."
        )
    return joblib.load(path), path

def yesno_to_int(v:str)->int:
    return 1 if str(v).strip().lower()=="yes" else 0

# Load the model once (no sidebar/settings)
try:
    pipe, model_used_path = load_model_auto()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------- patient form ----------
# st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section"><span style="color:#22D3EE;font-size:1.2rem;">〽️</span><div class="title">Patient Information</div></div>', unsafe_allow_html=True)

gender_list = ["Select gender","Male","Female","Other"]
yn_list     = ["Select","Yes","No"]
work_list   = ["Private","Self-employed","Govt_job","children","Never_worked"]
res_list    = ["Urban","Rural"]
smoke_list  = ["formerly smoked","never smoked","smokes","Unknown"]

with st.form("form", border=False):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        age_txt   = st.text_input("Age (years)", "", placeholder="e.g., 65")
        hyper_sel = st.selectbox("Hypertension", yn_list, 0)
        avg_txt   = st.text_input("Average Glucose Level (mg/dL)", "", placeholder="e.g., 120")
        work_type = st.selectbox("Work Type", work_list, 0)
        smoking   = st.selectbox("Smoking Status", smoke_list, 0)
    with c2:
        gender_sel= st.selectbox("Gender", gender_list, 0)
        heart_sel = st.selectbox("Heart Disease", yn_list, 0)
        bmi_txt   = st.text_input("BMI (kg/m²)", "", placeholder="e.g., 25.5")
        residence = st.selectbox("Residence Type", res_list, 0)

    submit = st.form_submit_button("Predict", use_container_width=True)

# ---------- predict ----------
THRESHOLD = 0.50

if submit:
    errs = []
    if gender_sel == "Select gender": errs.append("Please select a Gender.")
    if hyper_sel  == "Select":        errs.append("Please select Hypertension (Yes/No).")
    if heart_sel  == "Select":        errs.append("Please select Heart Disease (Yes/No).")
    try: age = float(age_txt.strip())
    except: errs.append("Age must be a number (e.g., 65).")
    try: avg = float(avg_txt.strip())
    except: errs.append("Average Glucose Level must be a number (e.g., 120).")
    try: bmi = float(bmi_txt.strip())
    except: errs.append("BMI must be a number (e.g., 25.5).")

    if errs:
        for e in errs: st.error(e)
    else:
        payload = {
            "gender": gender_sel, "age": age,
            "hypertension": yesno_to_int(hyper_sel),
            "heart_disease": yesno_to_int(heart_sel),
            "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": avg, "bmi": bmi, "smoking_status": smoking,
        }
        X = pd.DataFrame([payload], columns=EXPECTED_COLS)

        try:
            proba = float(pipe.predict_proba(X)[:, 1][0])
            pred  = int(proba >= THRESHOLD)
            label = "Positive (High Risk)" if pred else "Negative (Low Risk)"
            chip  = "bad" if pred else "ok"

            gval = max(0.0, min(1.0, proba))
            st.markdown(
                f"""
                <div class="metrics">
                  <div class="metric">
                    <div class="t">Predicted Probability</div>
                    <div class="gauge" style="--val:{gval};"><div class="c">{proba:.1%}</div></div>
                  </div>
                  <div class="metric">
                    <div class="t">Predicted Class</div>
                    <div class="v">{pred}</div>
                    <div class="chip {chip}">Decision: {label} • Threshold {THRESHOLD:.2f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander("Payload sent to model"):
                st.json(payload)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)  # end container

