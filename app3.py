import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Burn sample3.pdf (Outcome: 0=death, 1=survived)
# Features used: Burn% and Age ONLY (Sex intentionally ignored)
data = [
    # ---- Page 1 ----
    (100, 18, 0),
    (35, 44, 0),
    (90, 60, 0),
    (30, 31, 1),
    (75, 18, 0),
    (13, 6, 1),
    (60, 4, 0),
    (20, 36, 1),
    (10, 5, 1),
    (8, 0.75, 1),      # 9 months
    (20, 29, 1),
    (100, 32, 0),
    (15, 1.5, 1),
    (15, 17, 1),
    (10, 4, 1),
    (15, 25, 1),
    (100, 15, 0),
    (100, 35, 0),
    (35, 2.5, 1),
    (30, 30, 1),
    (50, 33, 1),
    (45, 2, 1),
    (70, 29, 0),
    (60, 40, 0),
    (50, 45, 0),
    (20, 27, 1),
    (35, 1.5, 1),
    (95, 18, 0),
    (50, 25, 0),
    (15, 20, 1),
    (10, 40, 1),
    (18, 39, 1),
    (20, 27, 1),

    # ---- Page 2 ----
    (60, 31, 1),
    (25, 26, 1),
    (40, 26, 1),
    (15, 20, 1),
    (30, 25, 1),
    (80, 16, 0),
    (20, 30, 1),
    (30, 30, 1),
    (10, 30, 1),
    (25, 1, 1),
    (65, 25, 0),
    (100, 26, 0),
    (35, 37, 0),
    (5, 2, 1),
    (70, 32, 0),
    (10, 5, 1),
    (40, 29, 1),
    (80, 32, 0),
    (100, 28, 0),
    (40, 24, 1),
    (50, 37, 0),
    (20, 1.5, 1),
    (35, 38, 1),
    (5, 11, 1),
    (100, 18, 0),
    (90, 45, 0),
    (20, 21, 1),
    (25, 28, 1),
    (40, 22, 1),
    (60, 3, 0),
    (50, 22, 0),
    (30, 12, 1),

    # ---- Page 3 ----
    (25, 28, 1),
    (20, 1.5, 1),
    (25, 19, 1),
    (40, 23, 1),
    (15, 2, 1),
    (30, 2.5, 0),
    (20, 23, 1),
    (7, 6, 1),
    (17, 40, 1),
    (15, 2, 1),
    (5, 5, 1),
    (10, 3, 1),
    (30, 15, 1),
    (25, 33, 1),
    (95, 17, 0),
    (60, 22, 0),
    (11, 74, 1),
    (15, 16, 1),
    (7, 3, 1),
    (95, 18, 0),
    (5, 5, 1),
    (90, 42, 0),
    (40, 24, 1),
    (30, 28, 1),
    (4, 6, 1),
    (10, 9, 1),
    (5, 1.5, 1),
    (10, 42, 1),
    (25, 27, 1),
    (10, 46, 1),
    (80, 27, 0),
    (10, 30, 1),
    (15, 2, 1),
    (10, 7, 1),
    (15, 16, 1),
    (7, 17, 1),
    (20, 19, 1),

    # ---- Page 4 ----
    (60, 18, 0),
    (100, 3, 0),
    (7, 1, 1),
    (55, 35, 1),
    (10, 0.75, 0),     # 9 months
    (70, 27, 0),
    (10, 13, 1),
    (30, 16, 0),
    (10, 4, 1),
    (50, 20, 0),
    (20, 9, 1),
    (9, 8, 1),
    (20, 32, 1),
    (40, 42, 1),
    (45, 11, 1),
]

df = pd.DataFrame(data, columns=["Burn", "Age", "Outcome"])
X = df[["Burn", "Age"]]
y = df["Outcome"].astype(int)

preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), ["Burn", "Age"])],
    remainder="drop"
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]
)

model.fit(X, y)

st.set_page_config(page_title="Burn Mortality Predictor", layout="centered")
st.title("Burn Mortality Predictor (Age + Burn% only)")

age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=25.0, step=0.5)
burn = st.slider("Burn % (TBSA)", min_value=0, max_value=100, value=20)

if st.button("Predict"):
    input_df = pd.DataFrame([{"Burn": float(burn), "Age": float(age)}])
    p_survive = float(model.predict_proba(input_df)[0, 1])
    p_die = 1.0 - p_survive

    st.subheader("Result")
    st.metric("Predicted mortality risk", f"{p_die*100:.2f}%")
    st.metric("Predicted survival probability", f"{p_survive*100:.2f}%")
    st.caption("Trained only on Burn sample3. Sex ignored by design.")
