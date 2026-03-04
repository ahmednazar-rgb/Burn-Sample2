import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")
st.title("Burn Mortality Predictor")

file = st.file_uploader("Upload burn data.xlsx", type=["xlsx"])

if file is not None:

    df = pd.read_excel(file, header=2)

    df.columns = [
        "Index","Name","Burn","Age","Sex","DOA","DOD","Outcome"
    ]

    df = df[["Burn","Age","Outcome"]]

    df["Outcome"] = df["Outcome"].astype(str).str.lower().str.strip()

    df = df[~df["Outcome"].isin(["u","unknown"])]

    df["Outcome"] = df["Outcome"].replace({
        "alive":1,
        "survived":1,
        "a":1,
        "1":1,
        "dead":0,
        "d":0,
        "0":0
    })

    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")

    df = df.dropna()

    st.write("Outcome distribution")
    st.write(df["Outcome"].value_counts())

    X = df[["Age","Burn"]]
    y = df["Outcome"]

    preprocess = ColumnTransformer([
        ("scale", StandardScaler(), ["Age","Burn"])
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000, C=0.5))
    ])

    model.fit(X,y)

    st.success(f"Model trained on {len(df)} patients")

    age = st.number_input("Age",0,120)
    burn = st.slider("Burn % TBSA",0,100)

    if st.button("Predict"):

        patient = pd.DataFrame([[age,burn]],columns=["Age","Burn"])

        survive = model.predict_proba(patient)[0][1]
        death = 1 - survive

        st.subheader("Prediction")

        st.write("Mortality Risk:", f"{death*100:.2f}%")
        st.write("Survival Probability:", f"{survive*100:.2f}%")
