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
st.write("Prediction based only on Age and Burn %. Non-linear effects and age groups included.")

uploaded_file = st.file_uploader("Upload burn data Excel file", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

df.columns = ["ID","Name","Burn","Age","Sex","DOA","DOD","Outcome"]

df["Outcome"] = df["Outcome"].astype(str).str.strip()
df = df[df["Outcome"].isin(["0","1"])]

df["Outcome"] = df["Outcome"].astype(int)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")

df = df.dropna(subset=["Age","Burn","Outcome"])

if df["Burn"].max() <= 1:
    df["Burn"] = df["Burn"] * 100

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
st.write("Deaths:", (df["Outcome"]==0).sum())
st.write("Survivals:", (df["Outcome"]==1).sum())

selected_group = st.selectbox(
    "Select age group model",
    ["Pediatric (<18)", "Adult (18–60)", "Elderly (>60)"]
)

group_df = df[df["AgeGroup"] == selected_group]

st.write("Patients in selected group:", len(group_df))

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
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),   # adds Age², Burn², Age×Burn
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000))
])

model.fit(X_train, y_train)

st.subheader("Prediction")

age = st.number_input("Age",0,120,30)
burn = st.slider("Burn %",0.0,100.0,30.0)

input_data = pd.DataFrame([[age,burn]],columns=["Age","Burn"])

prob_survival = model.predict_proba(input_data)[0][1]
prob_death = 1-prob_survival

st.write("Mortality risk:", round(prob_death*100,2), "%")
st.write("Survival probability:", round(prob_survival*100,2), "%")

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
st.write("Model AUC:", round(auc,3))
