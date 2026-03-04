# app.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

DEFAULT_XLSX_PATH = "/mnt/data/burn data.xlsx"

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")

st.title("Burn Center Mortality Predictor")
st.caption("Model predicts probability of death (mortality) using Age and Burn % only. Outcome: 0=death, 1=survival, U/u=unknown (excluded).")

@st.cache_data(show_spinner=False)
def load_and_clean_data(xlsx_path: str) -> pd.DataFrame:
    # File has a blank row and then a row containing labels inside the first data row.
    raw = pd.read_excel(xlsx_path, header=1)

    # First row still contains labels ("Name", "%", "Age"...). Drop it.
    raw = raw.dropna(how="all")
    raw = raw.iloc[1:].copy()

    # Rename expected columns (based on the attached sheet structure)
    raw.columns = ["idx", "name", "burn", "age", "sex", "doa", "dod", "outcome"]

    # Normalize outcome
    out = raw["outcome"].astype(str).str.strip()
    out = out.replace({"u": "U"})
    raw["outcome_norm"] = out

    # Keep only known outcomes 0/1
    known = raw[raw["outcome_norm"].isin(["0", "1"])].copy()

    # Convert features to numeric
    known["burn"] = pd.to_numeric(known["burn"], errors="coerce")
    known["age"] = pd.to_numeric(known["age"], errors="coerce")

    # Drop rows missing required fields
    known = known.dropna(subset=["burn", "age", "outcome_norm"]).copy()

    # Burn values in this sheet are fractions (0.04 to 1.0). Convert to percent 0-100.
    if known["burn"].max() <= 1.5:
        known["burn_pct"] = known["burn"] * 100.0
    else:
        known["burn_pct"] = known["burn"].astype(float)

    known["age_years"] = known["age"].astype(float)

    # y_death: 1 if death, 0 if survival
    known["y_death"] = (known["outcome_norm"] == "0").astype(int)

    return known[["age_years", "burn_pct", "y_death"]].reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X = df[["age_years", "burn_pct"]].copy()
    y = df["y_death"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

    model.fit(X_train, y_train)

    # Metrics
    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)

    y_hat = (p_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_hat)
    report = classification_report(y_test, y_hat, digits=3)

    return model, auc, cm, report, (X_train, X_test, y_train, y_test)

with st.sidebar:
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    use_default = st.checkbox(f"Use attached sheet at {DEFAULT_XLSX_PATH}", value=(uploaded is None))
    if uploaded is not None and not use_default:
        xlsx_bytes = uploaded.getvalue()
        tmp_path = "/mnt/data/_uploaded_burn_data.xlsx"
        with open(tmp_path, "wb") as f:
            f.write(xlsx_bytes)
        data_path = tmp_path
    else:
        data_path = DEFAULT_XLSX_PATH

df = load_and_clean_data(data_path)

if df.empty:
    st.error("No usable rows after cleaning. Ensure Outcome column contains 0/1 (U/u excluded) and Age/Burn are present.")
    st.stop()

model, auc, cm, report, splits = train_model(df)

col1, col2, col3 = st.columns(3)
col1.metric("Usable records", f"{len(df)}")
col2.metric("Deaths (0)", f"{int(df['y_death'].sum())}")
col3.metric("Survivals (1)", f"{int((1 - df['y_death']).sum())}")

st.divider()
st.subheader("Predict mortality")

age_min = int(max(0, np.floor(df["age_years"].min())))
age_max = int(np.ceil(df["age_years"].max()))
burn_min = float(max(0.0, np.floor(df["burn_pct"].min())))
burn_max = float(min(100.0, np.ceil(df["burn_pct"].max())))

age_in = st.number_input("Age (years)", min_value=0, max_value=max(120, age_max), value=int(np.clip(25, 0, max(120, age_max))), step=1)
burn_in = st.slider("Burn (%TBSA)", min_value=0.0, max_value=100.0, value=float(np.clip(30.0, 0.0, 100.0)), step=0.5)

x = pd.DataFrame({"age_years": [float(age_in)], "burn_pct": [float(burn_in)]})
p_death = float(model.predict_proba(x)[0, 1])
p_survival = 1.0 - p_death

st.write(f"**Predicted mortality (death probability):** {p_death*100:.1f}%")
st.write(f"**Predicted survival probability:** {p_survival*100:.1f}%")

st.divider()
st.subheader("Model quality (internal split)")
st.write(f"ROC AUC (test): **{auc:.3f}**")
st.text("Confusion matrix at 0.50 threshold (rows=true, cols=pred):")
st.write(pd.DataFrame(cm, index=["True:Survival", "True:Death"], columns=["Pred:Survival", "Pred:Death"]))
st.text("Classification report (test):")
st.text(report)

st.caption("This is a simple logistic model trained only on Age and Burn %. It is not a clinical decision tool.")
