import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")
st.title("Burn Mortality Predictor")

uploaded_file = st.file_uploader("Upload burn data Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:

    # Read Excel (headers start on row 3 in your sheet)
    df = pd.read_excel(uploaded_file, header=2)

    # Rename columns based on known sheet structure
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

    # Keep required variables
    df = df[["Burn", "Age", "Outcome"]]

    # Convert Outcome text to numeric
    df["Outcome"] = df["Outcome"].astype(str).str.lower().str.strip()

    df["Outcome"] = df["Outcome"].replace({
        "alive":1,
        "survived":1,
        "a":1,
        "1":1,

        "dead":0,
        "d":0,
        "0":0
    })

    # Remove unknown outcomes
    df = df[df["Outcome"] != "u"]

    # Convert numeric columns
    df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")

    df = df.dropna()

    # Create Baux score
    df["Baux"] = df["Age"] + df["Burn"]

    st.write("Outcome distribution:")
    st.write(df["Outcome"].value_counts())

    # Train model
    X = df[["Age", "Burn", "Baux"]]
    y = df["Outcome"]

    preprocess = ColumnTransformer(
        [("scale", StandardScaler(), ["Age", "Burn", "Baux"])]
    )

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    model.fit(X, y)

    st.success(f"Model trained on {len(df)} patients")

    # Prediction interface
    age = st.number_input("Age (years)", 0, 120)
    burn = st.slider("Burn % TBSA", 0, 100)

    if st.button("Predict Mortality"):

        baux = age + burn

        patient = pd.DataFrame(
            [[age, burn, baux]],
            columns=["Age", "Burn", "Baux"]
        )

        survive_prob = model.predict_proba(patient)[0][1]
        death_prob = 1 - survive_prob

        st.subheader("Prediction")

        st.write("Baux Score:", baux)
        st.write("Mortality Risk:", round(death_prob * 100, 2), "%")
        st.write("Survival Probability:", round(survive_prob * 100, 2), "%")
