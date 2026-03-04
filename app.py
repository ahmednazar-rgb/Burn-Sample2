import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Burn Mortality Predictor")

uploaded_file = st.file_uploader("Upload burn dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    df = df.iloc[2:].reset_index(drop=True)
    df.columns = ["x","Name","Burn","Age","Sex","DOA","DOD","Outcome"]

    df = df[["Burn","Age","Outcome"]]

    df = df[df["Outcome"]!="U"]

    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce") * 100
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")

    df = df.dropna()

    X = df[["Burn","Age"]]
    y = df["Outcome"]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), ["Burn","Age"])
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=3000))
    ])

    model.fit(X,y)

    st.success("Dataset loaded and model trained")

    age = st.number_input("Age (years)",0.0,120.0)
    burn = st.slider("Burn %",0,100)

    if st.button("Predict"):
        patient = pd.DataFrame([[burn,age]],columns=["Burn","Age"])

        survive = model.predict_proba(patient)[0][1]
        death = 1 - survive

        st.write("Mortality Risk:", round(death*100,2), "%")
        st.write("Survival Probability:", round(survive*100,2), "%")
