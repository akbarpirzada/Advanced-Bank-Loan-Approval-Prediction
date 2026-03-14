######################################################################################################
# Importing Libraries
######################################################################################################
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

######################################################################################################
# Streamlit Page Setup
######################################################################################################
st.set_page_config(page_title="💵 Loan Approval Prediction", layout="wide", initial_sidebar_state="expanded")

######################################################################################################
# Custom CSS
######################################################################################################
st.markdown("""
<style>
    /* ── Global font & background ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Main title ── */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-caption {
        color: #6c757d;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 1.5rem;
    }

    /* ── Metric card ── */
    .metric-card {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        border-radius: 14px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(26,115,232,0.25);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-card .metric-label {
        font-size: 0.82rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a73e8;
        border-left: 4px solid #1a73e8;
        padding-left: 10px;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* ── Input group labels ── */
    .input-group-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* ── Confusion matrix cells ── */
    .cm-table th, .cm-table td {
        text-align: center !important;
    }

    /* ── Result banners ── */
    .result-approved {
        background: linear-gradient(135deg, #00c853, #009624);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,200,83,0.35);
        animation: pop 0.4s ease;
    }
    .result-rejected {
        background: linear-gradient(135deg, #f44336, #b71c1c);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(244,67,54,0.35);
        animation: pop 0.4s ease;
    }
    .result-icon  { font-size: 3rem; }
    .result-name  { font-size: 1.6rem; font-weight: 700; margin: 6px 0; }
    .result-label { font-size: 1.1rem; opacity: 0.9; }
    .result-prob  { font-size: 2.2rem; font-weight: 800; margin-top: 8px; }

    @keyframes pop {
        0%   { transform: scale(0.92); opacity: 0; }
        100% { transform: scale(1);    opacity: 1; }
    }

    /* ── Probability bar ── */
    .prob-bar-wrap {
        background: #e9ecef;
        border-radius: 50px;
        height: 18px;
        overflow: hidden;
        margin: 10px 0 4px;
    }
    .prob-bar-fill-approved {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #00c853, #009624);
        transition: width 0.8s ease;
    }
    .prob-bar-fill-rejected {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #f44336, #b71c1c);
        transition: width 0.8s ease;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d47a1 0%, #1a73e8 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: #0d47a1 !important;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        width: 100%;
        padding: 0.5rem 1rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e8f0fe;
    }
    [data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        color: white !important;
        border-radius: 8px;
    }

    /* ── Divider ── */
    hr { border-color: #e9ecef !important; margin: 1.5rem 0; }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

######################################################################################################
# Header
######################################################################################################
#st.markdown('<p class="main-title">💵 Loan Approval Prediction</p>', unsafe_allow_html=True)
st.title("💵 Loan Approval Prediction")
st.markdown('<p class="sub-caption">Machine Learning · Logistic Regression · Pakistani Loan Dataset — for practice purposes only</p>', unsafe_allow_html=True)

######################################################################################################
# Data Loading (cached)
######################################################################################################
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

######################################################################################################
# Model Training (cached)
######################################################################################################
@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "approved"
    drop_cols = [target]
    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    X = df.drop(columns=drop_cols)
    y = df[target]

    cat_cols = [c for c in ["gender", "city", "employment_type", "bank"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return clf, metrics, X_train.columns.tolist()

######################################################################################################
# Sidebar — (1) Load Dataset
######################################################################################################
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")
st.sidebar.markdown("**① Load Dataset**")

csv_path = st.sidebar.text_input(
    "CSV Path",
    value="loan_dataset.csv",
    help="Path to your dataset CSV file."
)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"❌ Could not load CSV: {e}")
    st.stop()

total_rows   = len(df)
approved_cnt = int(df["approved"].sum()) if "approved" in df.columns else 0
rejected_cnt = total_rows - approved_cnt

st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.15);border-radius:10px;padding:12px 16px;margin-top:8px;">
  <div style="font-size:0.82rem;opacity:0.8;">DATASET SUMMARY</div>
  <div style="font-size:1.4rem;font-weight:700;">{total_rows:,} rows</div>
  <div style="font-size:0.85rem;margin-top:4px;">✅ Approved: {approved_cnt:,}</div>
  <div style="font-size:0.85rem;">❌ Rejected: {rejected_cnt:,}</div>
</div>
""", unsafe_allow_html=True)

######################################################################################################
# Sidebar — (2) Train Model
######################################################################################################
st.sidebar.markdown("---")
st.sidebar.markdown("**② Train Model**")

if st.sidebar.button("🚀 Train / Re-Train Model"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared — retraining…")

with st.spinner("Training model…"):
    clf, metrics, feature_order = train_model(df)

st.sidebar.success("✅ Model ready!")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.75rem;opacity:0.7;text-align:center;'>Developed by Akbar Pirzada | AI & ML Developer</div>",
    unsafe_allow_html=True
)

######################################################################################################
# Section 1 — Metrics
######################################################################################################
st.markdown('<div class="section-header">📊 Model Performance (Holdout Test Set)</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metric_items = [
    (m1, "Accuracy",  metrics["accuracy"],  "🎯"),
    (m2, "Precision", metrics["precision"], "🔍"),
    (m3, "Recall",    metrics["recall"],    "📡"),
    (m4, "F1 Score",  metrics["f1"],        "⚖️"),
]
for col, label, value, icon in metric_items:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem;">{icon}</div>
            <div class="metric-value">{value:.2%}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

######################################################################################################
# Section 2 — Data Preview + Confusion Matrix
######################################################################################################
st.markdown("<br>", unsafe_allow_html=True)
colA, colB = st.columns([1.6, 1])

with colA:
    st.markdown('<div class="section-header">🗂️ Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=280)

with colB:
    st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted: No ❌", "Predicted: Yes ✅"],
        index=["Actual: No ❌", "Actual: Yes ✅"]
    )
    st.dataframe(cm_df, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    <div style="font-size:0.82rem;color:#555;line-height:1.8;margin-top:8px;">
        <b>TN</b> {tn} &nbsp;|&nbsp; <b>FP</b> {fp}<br>
        <b>FN</b> {fn} &nbsp;|&nbsp; <b>TP</b> {tp}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

######################################################################################################
# Section 3 — Try a Prediction
######################################################################################################
st.markdown('<div class="section-header">🧪 Try a Prediction</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Personal Info ──────────────────────────────────────────────────────
st.markdown('<div class="input-group-label">👤 Personal Information</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)

with p1:
    applicant_name = st.text_input("Applicant Name", value="Muhammad Ali")
with p2:
    gender = st.selectbox("Gender", ["M", "F"], index=0, format_func=lambda x: "Male" if x == "M" else "Female")
with p3:
    age = st.slider("Age", 21, 60, 30)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Location & Employment ─────────────────────────────────────────────
st.markdown('<div class="input-group-label">🏢 Location & Employment</div>', unsafe_allow_html=True)
e1, e2, e3 = st.columns(3)

with e1:
    city = st.selectbox("City", sorted(df["city"].unique().tolist()))
with e2:
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique().tolist()))
with e3:
    bank = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Financials ─────────────────────────────────────────────────────────
st.markdown('<div class="input-group-label">💰 Financial Details</div>', unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns(5)

with f1:
    monthly_income_pkr = st.number_input("Monthly Income (PKR)", min_value=1_500, max_value=500_000, value=120_000, step=1_000)
with f2:
    credit_score = st.slider("Credit Score", 300, 900, 680)
with f3:
    loan_amount_pkr = st.number_input("Loan Amount (PKR)", min_value=50_000, max_value=3_500_000, value=800_000, step=5_000)
with f4:
    loan_tenure_months = st.selectbox("Tenure (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
with f5:
    existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3], index=0)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 4: History ────────────────────────────────────────────────────────────
st.markdown('<div class="input-group-label">📋 Credit History</div>', unsafe_allow_html=True)
h1, h2, _ = st.columns([1, 1, 2])

with h1:
    default_history = st.selectbox("Default History", [0, 1],
                                   format_func=lambda x: "No Default ✅" if x == 0 else "Has Defaulted ⚠️", index=0)
with h2:
    has_credit_card = st.selectbox("Has Credit Card", [0, 1],
                                   format_func=lambda x: "No 💳" if x == 0 else "Yes 💳", index=0)

######################################################################################################
# Build Input Row
######################################################################################################
input_row = pd.DataFrame([{
    "gender":             gender,
    "age":                age,
    "city":               city,
    "employment_type":    employment_type,
    "bank":               bank,
    "monthly_income_pkr": monthly_income_pkr,
    "credit_score":       credit_score,
    "loan_amount_pkr":    loan_amount_pkr,
    "loan_tenure_months": loan_tenure_months,
    "existing_loans":     existing_loans,
    "default_history":    default_history,
    "has_credit_card":    has_credit_card
}])
input_row = input_row[feature_order]

######################################################################################################
# Predict Button + Result
######################################################################################################
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict_clicked = st.button("🔮 Predict Approval", use_container_width=True)

if predict_clicked:
    prob = float(clf.predict_proba(input_row)[:, 1][0])
    pred = int(prob >= 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 2, 1])

    with r2:
        if pred == 1:
            bar_class = "prob-bar-fill-approved"
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-name">{applicant_name}</div>
                <div class="result-label">Loan Application</div>
                <div class="result-prob">APPROVED</div>
                <div style="margin-top:16px;">
                    <div style="font-size:0.85rem;opacity:0.85;margin-bottom:6px;">Approval Probability</div>
                    <div class="prob-bar-wrap">
                        <div class="{bar_class}" style="width:{prob*100:.1f}%;"></div>
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;">{prob:.2%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bar_class = "prob-bar-fill-rejected"
            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-name">{applicant_name}</div>
                <div class="result-label">Loan Application</div>
                <div class="result-prob">REJECTED</div>
                <div style="margin-top:16px;">
                    <div style="font-size:0.85rem;opacity:0.85;margin-bottom:6px;">Approval Probability</div>
                    <div class="prob-bar-wrap">
                        <div class="{bar_class}" style="width:{prob*100:.1f}%;"></div>
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;">{prob:.2%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Key factors summary ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Application Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Credit Score",        credit_score, delta=None)
    s2.metric("Monthly Income (PKR)", f"{monthly_income_pkr:,}")
    s3.metric("Loan Amount (PKR)",    f"{loan_amount_pkr:,}")
    s4.metric("Existing Loans",       existing_loans)