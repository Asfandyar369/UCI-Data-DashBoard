import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

st.set_page_config(
    page_title="CareLens | Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Visual theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg,#f5f8ff 0%,#f8fbff 55%,#f1f7f7 100%); }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
.hero {
  background: linear-gradient(115deg,#102a43 0%,#176b87 58%,#2a9d8f 100%);
  padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 18px;
  box-shadow: 0 12px 30px rgba(16,42,67,.16);
}
.hero h1 { color:white; font-size:2.15rem; margin:0 0 6px 0; font-weight:800; }
.hero p { color:#e5f4f5; margin:0; font-size:1rem; }
.kpi {
  background: white; border: 1px solid #e5edf5; border-radius: 18px;
  padding: 18px 20px; box-shadow: 0 5px 18px rgba(26,54,93,.06);
  min-height: 112px;
}
.kpi-label { color:#62748a; font-size:.84rem; font-weight:600; }
.kpi-value { color:#102a43; font-size:1.8rem; font-weight:800; margin-top:4px; }
.kpi-note { color:#8292a6; font-size:.76rem; margin-top:2px; }
.panel {
  background:white; border:1px solid #e5edf5; border-radius:18px;
  padding:18px 20px; box-shadow:0 5px 18px rgba(26,54,93,.05);
}
.section-title { color:#102a43; font-weight:800; font-size:1.12rem; margin:4px 0 2px 0; }
.section-sub { color:#718096; font-size:.84rem; margin-bottom:12px; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #e5edf5; padding:14px 16px;
  border-radius:16px; box-shadow:0 4px 14px rgba(26,54,93,.05);
}
div[data-testid="stTabs"] button { font-weight:700; }
div[data-testid="stAlert"] { border-radius:14px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏥 CareLens — Healthcare Readmission Analytics</h1>
  <p>Explore hospital encounter patterns, evaluate a machine-learning model, and support human-centred healthcare decisions.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Dashboard controls")
    uploaded = st.file_uploader("Upload diabetic_data.csv", type=["csv"])
    st.caption("Expected UCI Diabetes 130-US Hospitals dataset.")
    st.divider()
    test_pct = st.slider("Test data (%)", 10, 40, 20, 5)
    threshold = st.slider("Prediction threshold", 0.10, 0.90, 0.50, 0.05)
    st.divider()
    st.markdown("### 🧭 Navigation")
    st.caption("Use the tabs in the main dashboard to explore insights.")
    st.info("Historical data: 1999–2008\n\nEducational prototype — not for clinical use.")

if uploaded is None:
    st.markdown("""
    <div class="panel">
      <h3>👋 Welcome to CareLens</h3>
      <p>Upload your <b>diabetic_data.csv</b> file from the sidebar to open the interactive dashboard.</p>
      <p>You'll get:</p>
      <ul>
        <li>Visual readmission breakdowns and patient-encounter patterns</li>
        <li>Age-group and hospital-stay comparisons</li>
        <li>Logistic Regression performance, ROC curve and confusion matrix</li>
        <li>Interactive sample-encounter prediction</li>
        <li>Industry 4.0 / Industry 5.0 healthcare insights</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)

try:
    df = load_data(uploaded.getvalue()).copy()
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

required = ["readmitted", "age", "time_in_hospital", "number_diagnoses"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Required columns are missing: " + ", ".join(missing))
    st.stop()

df["readmitted"] = df["readmitted"].astype(str).str.strip().str.upper()
df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()
if df.empty:
    st.error("No valid readmission categories found. Expected <30, >30 and NO.")
    st.stop()

df["early_readmission"] = (df["readmitted"] == "<30").astype(int)
for c in ["time_in_hospital", "number_diagnoses"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["age"] = df["age"].astype(str)
df["age"] = df["age"].replace({"nan": "Unknown", "None": "Unknown"})
total = len(df)
early = int(df["early_readmission"].sum())
rate = early / total * 100
not_early = total - early
avg_days = df["time_in_hospital"].mean()
avg_dx = df["number_diagnoses"].mean()

# ---------- Model ----------
X = df[["age", "time_in_hospital", "number_diagnoses"]].copy()
y = df["early_readmission"]
preprocess = ColumnTransformer([
    ("age", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), ["age"]),
    ("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), ["time_in_hospital", "number_diagnoses"])
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

if "patient_nbr" in df.columns and df["patient_nbr"].notna().any():
    groups = df["patient_nbr"].fillna(-1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_pct / 100, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    split_note = "Patient-level split: records for the same patient are kept in one partition."
else:
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_pct / 100, random_state=42, stratify=y
    )
    split_note = "Stratified row-level split (patient_nbr unavailable)."

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
if y_train.nunique() < 2 or y_test.nunique() < 2:
    st.error("The train/test split contains only one target class. Change the test percentage.")
    st.stop()

with st.spinner("Training the Logistic Regression model..."):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)

# ---------- KPI strip ----------
kpis = [
    ("🏨", "Hospital encounters", f"{total:,}", "Rows in the uploaded dataset"),
    ("⚠️", "Early readmissions", f"{early:,}", "Readmitted within 30 days"),
    ("📈", "Early readmission rate", f"{rate:.1f}%", "Share of valid encounters"),
    ("🛏️", "Average stay", f"{avg_days:.1f} days", "Across valid encounters"),
]
cols = st.columns(4)
for col, (icon, label, value, note) in zip(cols, kpis):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{icon} {label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )

st.write("")
overview, patterns, performance, predict, industry = st.tabs([
    "📊 Overview", "🔎 Patient patterns", "🤖 Model insights",
    "🧑‍⚕️ Try a prediction", "🌐 Industry 4.0 / 5.0"
])

# ---------- Overview ----------
with overview:
    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.markdown('<div class="section-title">Readmission outcome mix</div><div class="section-sub">How encounters are distributed across the three recorded categories.</div>', unsafe_allow_html=True)
        counts = df["readmitted"].value_counts().reindex(["<30", ">30", "NO"]).fillna(0).astype(int)
        donut = pd.DataFrame({
            "Outcome": ["Within 30 days", "After 30 days", "No readmission"],
            "Encounters": counts.values
        })
        fig = px.pie(donut, names="Outcome", values="Encounters", hole=.64,
                     color="Outcome",
                     color_discrete_map={"Within 30 days":"#ef6a6a","After 30 days":"#f4b942","No readmission":"#2a9d8f"})
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=3)))
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                          annotations=[dict(text=f"{total:,}<br><sup>encounters</sup>", x=.5, y=.5,
                                            font_size=18, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Outcome counts</div><div class="section-sub">Compare the number of encounters in each readmission category.</div>', unsafe_allow_html=True)
        bar = px.bar(donut, x="Outcome", y="Encounters", color="Outcome",
                     color_discrete_map={"Within 30 days":"#ef6a6a","After 30 days":"#f4b942","No readmission":"#2a9d8f"},
                     text_auto=".2s")
        bar.update_layout(showlegend=False, yaxis_title="Number of encounters", xaxis_title="",
                          margin=dict(t=10,b=10,l=10,r=10), plot_bgcolor="white")
        bar.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(bar, use_container_width=True)

    st.markdown('<div class="section-title">Dataset snapshot</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Dataset columns", f"{df.shape[1]:,}")
    s2.metric("Training encounters", f"{len(train_idx):,}")
    s3.metric("Testing encounters", f"{len(test_idx):,}")
    st.caption(split_note)
    with st.expander("Preview first 10 records"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# ---------- Patterns ----------
with patterns:
    st.markdown('<div class="section-title">Explore factors associated with early readmission</div><div class="section-sub">These are descriptive patterns, not proof that a factor causes readmission.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
                     "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        age_counts = df.groupby(["age", "early_readmission"]).size().reset_index(name="Encounters")
        age_counts["Outcome"] = age_counts["early_readmission"].map({0:"Not <30-day readmission",1:"<30-day readmission"})
        fig_age = px.bar(age_counts, x="age", y="Encounters", color="Outcome", barmode="group",
                         category_orders={"age": age_order},
                         color_discrete_map={"Not <30-day readmission":"#3b82a0","<30-day readmission":"#ef6a6a"})
        fig_age.update_layout(title="Encounter counts by age group", xaxis_title="Age group",
                              yaxis_title="Encounters", plot_bgcolor="white", margin=dict(t=45,b=10))
        st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        stay = df.dropna(subset=["time_in_hospital"])
        fig_stay = px.box(stay, x="early_readmission", y="time_in_hospital",
                          color="early_readmission", points=False,
                          color_discrete_map={0:"#3b82a0",1:"#ef6a6a"},
                          labels={"early_readmission":"Early readmission flag","time_in_hospital":"Hospital stay (days)"})
        fig_stay.update_layout(title="Hospital-stay duration by outcome", xaxis=dict(
            tickmode="array", tickvals=[0,1], ticktext=["Not <30 days","<30 days"]),
            plot_bgcolor="white", showlegend=False, margin=dict(t=45,b=10))
        st.plotly_chart(fig_stay, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        fig_dx = px.histogram(df, x="number_diagnoses", color="early_readmission",
                              barmode="overlay", opacity=.72, nbins=16,
                              color_discrete_map={0:"#3b82a0",1:"#ef6a6a"},
                              labels={"number_diagnoses":"Number of diagnoses","early_readmission":"Early readmission"})
        fig_dx.update_layout(title="Number of diagnoses distribution", plot_bgcolor="white",
                             margin=dict(t=45,b=10))
        st.plotly_chart(fig_dx, use_container_width=True)
    with c4:
        compare = df.groupby("early_readmission").agg(
            Average_stay=("time_in_hospital","mean"),
            Average_diagnoses=("number_diagnoses","mean"),
            Encounters=("early_readmission","size")
        ).reset_index()
        compare["Group"] = compare["early_readmission"].map({0:"Not <30 days",1:"<30 days"})
        long = compare.melt(id_vars="Group", value_vars=["Average_stay","Average_diagnoses"],
                            var_name="Measure", value_name="Average")
        fig_compare = px.bar(long, x="Measure", y="Average", color="Group", barmode="group",
                             color_discrete_map={"Not <30 days":"#3b82a0","<30 days":"#ef6a6a"},
                             labels={"Measure":"","Average":"Average value"})
        fig_compare.update_layout(title="Average stay and diagnoses by group", plot_bgcolor="white",
                                  margin=dict(t=45,b=10))
        st.plotly_chart(fig_compare, use_container_width=True)
    st.info("Interpretation: compare groups to identify patterns for further investigation. The dataset is historical and these associations should not be treated as clinical conclusions.")

# ---------- Model ----------
with performance:
    st.markdown('<div class="section-title">Model evaluation at the selected threshold</div><div class="section-sub">Logistic Regression using age group, hospital-stay duration and number of diagnoses.</div>', unsafe_allow_html=True)
    mc = st.columns(5)
    for col, label, val in zip(mc, ["Accuracy","Precision","Recall","F1-score","ROC-AUC"],
                               [acc,prec,rec,f1,auc]):
        col.metric(label, f"{val:.3f}")
    left, right = st.columns(2, gap="large")
    cm = confusion_matrix(y_test, preds, labels=[0,1])
    with left:
        heat = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: No","Predicted: Yes"], y=["Actual: No","Actual: Yes"],
            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        heat.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="",
                           margin=dict(t=45,b=10,l=10,r=10))
        st.plotly_chart(heat, use_container_width=True)
    with right:
        fpr, tpr, _ = roc_curve(y_test, probs)
        rocfig = go.Figure()
        rocfig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC={auc:.3f})",
                                    line=dict(width=3, color="#176b87"), fill="tozeroy"))
        rocfig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline",
                                    line=dict(dash="dash", color="#9aa7b5")))
        rocfig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                             yaxis_title="True positive rate", xaxis=dict(range=[0,1]),
                             yaxis=dict(range=[0,1]), margin=dict(t=45,b=10))
        st.plotly_chart(rocfig, use_container_width=True)
    report = classification_report(y_test, preds, labels=[0,1],
        target_names=["Not <30 days","<30 days"], output_dict=True, zero_division=0)
    st.markdown("**Detailed classification report**")
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.warning("Model scores are for learning and comparison. Accuracy alone is insufficient; false negatives and recall also matter.")

# ---------- Prediction ----------
with predict:
    st.markdown('<div class="section-title">Interactive sample encounter</div><div class="section-sub">Change the inputs to see how the trained model responds. This is not a patient-care recommendation.</div>', unsafe_allow_html=True)
    age_options = sorted(df["age"].dropna().astype(str).unique().tolist())
    default_age = "[60-70)" if "[60-70)" in age_options else age_options[0]
    with st.form("prediction_form"):
        p1,p2,p3 = st.columns(3)
        with p1:
            age = st.selectbox("Age group", age_options, index=age_options.index(default_age))
        with p2:
            days = st.slider("Time in hospital (days)", 1, 14, 5)
        with p3:
            diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
        submitted = st.form_submit_button("✨ Estimate model score", type="primary", use_container_width=True)
    if submitted:
        case = pd.DataFrame({"age":[str(age)],"time_in_hospital":[days],"number_diagnoses":[diagnoses]})
        probability = float(model.predict_proba(case)[0,1])
        flagged = probability >= threshold
        left, right = st.columns([1,1.25], gap="large")
        with left:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability*100,
                number={"suffix":"%","font":{"size":34}},
                title={"text":"Model score for <30-day category"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#ef6a6a" if flagged else "#2a9d8f"},
                       "steps":[{"range":[0,threshold*100],"color":"#e8f4f1"},
                                {"range":[threshold*100,100],"color":"#fff0ed"}],
                       "threshold":{"line":{"color":"#102a43","width":3},"thickness":.8,"value":threshold*100}}
            ))
            gauge.update_layout(height=300, margin=dict(t=55,b=15,l=20,r=20))
            st.plotly_chart(gauge, use_container_width=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### Model output")
            st.metric("Selected threshold", f"{threshold*100:.0f}%")
            if flagged:
                st.error("Flagged by the model at the selected threshold.")
            else:
                st.success("Not flagged by the model at the selected threshold.")
            st.markdown(f"- Age group: **{age}**\n- Hospital stay: **{days} days**\n- Number of diagnoses: **{diagnoses}**")
            st.markdown("</div>", unsafe_allow_html=True)
        st.warning("This is an uncalibrated educational model score based on only three features. It is not a diagnosis or a validated clinical risk estimate. A qualified healthcare professional must assess the full patient context.")

# ---------- Industry ----------
with industry:
    st.markdown('<div class="section-title">From data to human-centred care</div><div class="section-sub">How analytics can fit into an Industry 4.0 / Industry 5.0 healthcare workflow.</div>', unsafe_allow_html=True)
    steps = [
        ("📥","Hospital data","Encounter history and relevant records are collected."),
        ("🧠","AI analysis","A model identifies patterns and produces a review signal."),
        ("🔔","Care-team review","The signal can prompt a closer look at discharge and follow-up needs."),
        ("🤝","Human decision","Clinicians consider the complete patient situation and decide care.")
    ]
    cols = st.columns(4)
    for col,(ico,title,desc) in zip(cols,steps):
        col.markdown(f'<div class="kpi" style="min-height:170px"><div style="font-size:2rem">{ico}</div><div style="font-weight:800;color:#102a43;margin:8px 0">{title}</div><div style="font-size:.85rem;color:#718096">{desc}</div></div>',unsafe_allow_html=True)
    st.write("")
    a,b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel"><h3>⚙️ Industry 4.0</h3><ul><li>Digitized hospital records</li><li>Data analytics and machine learning</li><li>Faster identification of patterns</li><li>Evidence to support operational planning</li></ul></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><h3>❤️ Industry 5.0</h3><ul><li>Human-centred and personalized care</li><li>Clinician oversight and accountability</li><li>Transparency about model limitations</li><li>AI assists people; it does not replace them</li></ul></div>',unsafe_allow_html=True)
    st.success("Key message: AI can support healthcare professionals, but the final patient-care decision must remain with qualified professionals.")

with st.expander("Technical details"):
    st.write("Input features:", ["age", "time_in_hospital", "number_diagnoses"])
    st.write("Target: 1 when readmitted == '<30'; 0 for '>30' or 'NO'.")
    st.write("Training rows:", len(train_idx), "| Testing rows:", len(test_idx))
    st.write(split_note)

st.caption("CareLens educational dashboard • UCI Diabetes 130-US Hospitals dataset (1999–2008) • Not for clinical use.")
