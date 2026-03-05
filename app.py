# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Burn Center Mortality Predictor", layout="centered")

st.title("Burn Center Mortality Predictor")
st.caption("Features: Age + Burn %. Outcome: 0=death, 1=survival, U/u=unknown (excluded).")

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _find_col(cols, keys):
    # cols: list of column names, keys: list of normalized tokens to match
    ncols = {c: _norm(c) for c in cols}
    for c, nc in ncols.items():
        for k in keys:
            if k in nc:
                return c
    return None

@st.cache_data(show_spinner=False)
def load_burn_excel(uploaded_file) -> pd.DataFrame:
    # Read with best effort (some sheets have an extra label row)
    df0 = pd.read_excel(uploaded_file, header=0)
    df1 = pd.read_excel(uploaded_file, header=1)

    # Choose the one that seems to contain outcome + age + burn columns
    def score(df):
        cols = list(df.columns)
        s = 0
        if _find_col(cols, ["outcome", "status", "result"]): s += 2
        if _find_col(cols, ["age", "ageyears"]): s += 2
        if _find_col(cols, ["burn", "tbsa", "percent"]): s += 2
        return s, df

    df = max([score(df0), score(df1)], key=lambda x: x[0])[1].copy()
    df = df.dropna(how="all")

    cols = list(df.columns)

    c_out = _find_col(cols, ["outcome", "status", "result"])
    c_age = _find_col(cols, ["age"])
    c_burn = _find_col(cols, ["burn", "tbsa", "percent"])

    if c_out is None or c_age is None or c_burn is None:
        raise ValueError(
            "Could not detect required columns. Ensure your sheet has columns for Age, Burn (or TBSA), and Outcome."
        )

    out = df[c_out].astype(str).str.strip()
    out = out.replace({"u": "U", "Unknown": "U", "unknown": "U"})
    df["_out"] = out

    # keep only known outcomes 0/1
    df = df[df["_out"].isin(["0", "1"])].copy()

    # numeric conversion
    df["_age"] = pd.to_numeric(df[c_age], errors="coerce")
    df["_burn_raw"] = pd.to_numeric(df[c_burn], errors="coerce")
    df = df.dropna(subset=["_age", "_burn_raw", "_out"]).copy()

    # burn: if looks like fraction (0-1), convert to percent
    if df["_burn_raw"].max() <= 1.5:
        df["_burn_pct"] = df["_burn_raw"] * 100.0
    else:
        df["_burn_pct"] = df["_burn_raw"].astype(float)

    # label for mortality model: 1=death, 0=survival
    df["y_death"] = (df["_out"] == "0").astype(int)

    cleaned = df[["_age", "_burn_pct", "y_death"]].rename(
        columns={"_age": "age_years", "_burn_pct": "burn_pct"}
    )
    return cleaned.reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def train_model(clean_df: pd.DataFrame):
    X = clean_df[["age_years", "burn_pct"]].copy()
    y = clean_df["y_death"].copy()

    # stratify only if both classes exist
    strat = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced" if y.nunique() == 2 else None,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    metrics = {}
    if y_test.nunique() == 2:
        p = model.predict_proba(X_test)[:, 1]  # prob death
        metrics["auc"] = float(roc_auc_score(y_test, p))
    else:
        metrics["auc"] = None

    p_all = model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    y_hat = (p_all >= 0.5).astype(int) if len(p_all) else np.array([])
    metrics["cm"] = confusion_matrix(y_test, y_hat) if len(y_hat) else None
    metrics["report"] = classification_report(y_test, y_hat, digits=3) if len(y_hat) else None

    return model, metrics

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload the Excel file to train the model.")
    st.stop()

try:
    clean_df = load_burn_excel(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

if clean_df.empty:
    st.error("After cleaning, there are no usable rows (only Outcome 0/1 with Age and Burn present are used).")
    st.stop()

st.subheader("Training data used")
c1, c2, c3 = st.columns(3)
c1.metric("Rows used", f"{len(clean_df)}")
c2.metric("Deaths (Outcome=0)", f"{int(clean_df['y_death'].sum())}")
c3.metric("Survivals (Outcome=1)", f"{int((1-clean_df['y_death']).sum())}")
st.dataframe(clean_df.head(20), use_container_width=True)

model, metrics = train_model(clean_df)

st.divider()
st.subheader("Prediction")

age_in = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
burn_in = st.slider("Burn (%TBSA)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)

x = pd.DataFrame({"age_years": [float(age_in)], "burn_pct": [float(burn_in)]})
p_death = float(model.predict_proba(x)[0, 1])
p_survival = 1.0 - p_death

st.write(f"**Mortality risk (death probability):** {p_death*100:.1f}%")
st.write(f"**Survival probability:** {p_survival*100:.1f}%")

st.divider()
st.subheader("Model check (internal split)")

if metrics["auc"] is not None:
    st.write(f"ROC AUC (test): **{metrics['auc']:.3f}**")
else:
    st.write("ROC AUC not available (test split has only one class).")

if metrics["cm"] is not None:
    cm = metrics["cm"]
    st.text("Confusion matrix at 0.50 threshold (rows=true, cols=pred):")
    st.write(pd.DataFrame(cm, index=["True:Survival", "True:Death"], columns=["Pred:Survival", "Pred:Death"]))

if metrics["report"] is not None:
    st.text("Classification report (test):")
    st.text(metrics["report"])

st.caption("Educational use only. Not a clinical decision tool.")
