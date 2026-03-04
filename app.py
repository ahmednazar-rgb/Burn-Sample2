import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")
st.title("Burn Mortality Predictor (Age, Burn%, Baux)")

uploaded = st.file_uploader("Upload Excel dataset (.xlsx)", type=["xlsx"])

if uploaded is not None:
    df = pd.read_excel(uploaded)

    # Adjust these column names if your Excel differs
    # Expected columns: Burn, Age, Outcome
    df = df[["Burn","Age","Outcome"]]

    # Remove unknown outcomes
    df = df[df["Outcome"] != "U"]

    # Convert to numeric
    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")

    df = df.dropna()

    # Compute Baux score
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

    model.fit(X, y)

    st.success(f"Model trained on {len(df)} patients")

    age = st.number_input("Age (years)", 0.0, 120.0)
    burn = st.slider("Burn % TBSA", 0, 100)

    if st.button("Predict"):
        baux = age + burn
        patient = pd.DataFrame([[age, burn, baux]], columns=["Age","Burn","Baux"])

        p_survive = model.predict_proba(patient)[0][1]
        p_die = 1 - p_survive

        st.subheader("Prediction")
        st.write("Baux score:", round(baux,1))
        st.write("Mortality Risk:", round(p_die*100,2), "%")
        st.write("Survival Probability:", round(p_survive*100,2), "%")
