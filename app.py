import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Burn Mortality Predictor")

file = st.file_uploader("Upload burn data Excel", type=["xlsx"])

if file is not None:

    df = pd.read_excel(file, header=2)

    # rename columns according to your sheet
    df.columns = [
        "Index",
        "Name",
        "Burn",
        "Age",
        "Sex",
        "DOA",
        "DOD",
        "Outcome"
    ]

    df = df[["Burn","Age","Outcome"]]

    df = df[df["Outcome"] != "U"]

    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")

    df = df.dropna()

    df["Baux"] = df["Age"] + df["Burn"]

    X = df[["Age","Burn","Baux"]]
    y = df["Outcome"]

    preprocess = ColumnTransformer([
        ("scale", StandardScaler(), ["Age","Burn","Baux"])
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    model.fit(X,y)

    st.success(f"Model trained on {len(df)} patients")

    age = st.number_input("Age",0,120)
    burn = st.slider("Burn %",0,100)

    if st.button("Predict"):

        baux = age + burn

        patient = pd.DataFrame([[age,burn,baux]],
                               columns=["Age","Burn","Baux"])

        survive = model.predict_proba(patient)[0][1]
        death = 1 - survive

        st.write("Baux Score:", baux)
        st.write("Mortality Risk:", round(death*100,2), "%")
        st.write("Survival Probability:", round(survive*100,2), "%")
