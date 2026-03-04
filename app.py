import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")
st.title("Burn Mortality Predictor")

file = st.file_uploader("Upload burn data Excel (.xlsx)", type=["xlsx"])

if file is not None:
    # Your sheet has headers on row 3
    df = pd.read_excel(file, header=2)

    # Force expected column names for this sheet structure
    df.columns = ["Index","Name","Burn","Age","Sex","DOA","DOD","Outcome"]

    df = df[["Burn","Age","Outcome"]].copy()

    # Show raw outcome values to verify what exists in the sheet
    st.write("Raw Outcome values detected:")
    st.write(df["Outcome"].dropna().unique())

    # Normalize outcome text
    df["Outcome"] = df["Outcome"].astype(str).str.strip().str.lower()

    mapping = {
        "alive":1,"survived":1,"survive":1,"a":1,"1":1,"yes":1,
        "dead":0,"died":0,"death":0,"d":0,"0":0,"no":0
    }

    df["Outcome"] = df["Outcome"].replace(mapping)

    # Remove unknown outcomes
    df = df[~df["Outcome"].isin(["u","unknown","nan"])]

    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = df.dropna()

    # Compute Baux score
    df["Baux"] = df["Age"] + df["Burn"]

    st.write("Outcome distribution after cleaning:")
    st.write(df["Outcome"].value_counts())

    # Stop if only one class remains
    if df["Outcome"].nunique() < 2:
        st.error("Dataset contains only one outcome class after cleaning. Model cannot train.")
        st.stop()

    X = df[["Age","Burn","Baux"]]
    y = df["Outcome"]

    preprocess = ColumnTransformer(
        [("scale", StandardScaler(), ["Age","Burn","Baux"])]
    )

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    model.fit(X, y)

    st.success(f"Model trained on {len(df)} patients")

    age = st.number_input("Age", 0, 120)
    burn = st.slider("Burn % TBSA", 0, 100)

    if st.button("Predict"):
        baux = age + burn
        patient = pd.DataFrame([[age,burn,baux]], columns=["Age","Burn","Baux"])
        p_survive = model.predict_proba(patient)[0][1]
        p_die = 1 - p_survive

        st.subheader("Prediction")
        st.write("Baux Score:", baux)
        st.write("Mortality Risk:", round(p_die*100,2), "%")
        st.write("Survival Probability:", round(p_survive*100,2), "%")
