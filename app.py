import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Burn Mortality Predictor")

st.title("Burn Center Mortality Predictor")
st.write("Prediction based on Age and Burn %. Models separated by age group.")

uploaded_file = st.file_uploader("Upload burn data Excel file", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

# Rename columns
df.columns = ["ID","Name","Burn","Age","Sex","DOA","DOD","Outcome"]

# Clean outcome
df["Outcome"] = df["Outcome"].astype(str).str.strip()
df = df[df["Outcome"].isin(["0","1"])]
df["Outcome"] = df["Outcome"].astype(int)

# Convert numeric columns
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")

# Convert Excel fraction percentages to real percent
df["Burn"] = df["Burn"] * 100

# Drop invalid rows
df = df.dropna(subset=["Age","Burn","Outcome"])

# Keep logical ranges
df = df[(df["Burn"] >= 0) & (df["Burn"] <= 100)]
df = df[(df["Age"] >= 0) & (df["Age"] <= 120)]

# Age group function
def age_group(age):
    if age < 18:
        return "Pediatric (<18)"
    elif age <= 60:
        return "Adult (18–60)"
    else:
        return "Elderly (>60)"

df["AgeGroup"] = df["Age"].apply(age_group)

st.subheader("Dataset Summary")
st.write("Total patients:", len(df))
st.write("Deaths:", (df["Outcome"] == 0).sum())
st.write("Survivals:", (df["Outcome"] == 1).sum())

selected_group = st.selectbox(
    "Select age group model",
    ["Pediatric (<18)", "Adult (18–60)", "Elderly (>60)"]
)

group_df = df[df["AgeGroup"] == selected_group].copy()

st.write("Patients in selected group:", len(group_df))

if len(group_df) < 10:
    st.error("Not enough data in this age group.")
    st.stop()

if group_df["Outcome"].nunique() < 2:
    st.error("Only one outcome present. Model cannot train.")
    st.stop()

X = group_df[["Age","Burn"]]
y = group_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000))
])

model.fit(X_train, y_train)

if selected_group == "Pediatric (<18)":
    age_min, age_max = 0, 17
elif selected_group == "Adult (18–60)":
    age_min, age_max = 18, 60
else:
    age_min, age_max = 61, 120

st.subheader("Prediction")

age = st.slider(
    "Age",
    min_value=age_min,
    max_value=age_max,
    value=age_min
)

burn = st.slider(
    "Burn %",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

input_data = pd.DataFrame([[age, burn]], columns=["Age","Burn"])

prob_survival = model.predict_proba(input_data)[0][1]
prob_death = 1 - prob_survival

st.write("Mortality risk:", round(prob_death * 100, 2), "%")
st.write("Survival probability:", round(prob_survival * 100, 2), "%")

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
st.write("Model AUC:", round(auc, 3))
