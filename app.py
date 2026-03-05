import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Burn Mortality Predictor")

st.title("Burn Center Mortality Predictor")
st.write("Predicts mortality using Age and Burn %.")

uploaded_file = st.file_uploader("Upload burn data Excel file", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

# Rename columns from the sheet
df.columns = ["ID","Name","Burn","Age","Sex","DOA","DOD","Outcome"]

# Remove unknown outcomes
df["Outcome"] = df["Outcome"].astype(str).str.strip()
df = df[df["Outcome"].isin(["0","1"])]

# Convert types
df["Outcome"] = df["Outcome"].astype(int)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Burn"] = pd.to_numeric(df["Burn"], errors="coerce")

df = df.dropna(subset=["Age","Burn","Outcome"])

# Convert burn fraction to %
if df["Burn"].max() <= 1:
    df["Burn"] = df["Burn"] * 100

X = df[["Age","Burn"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25,random_state=42
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

model.fit(X_train,y_train)

st.subheader("Dataset Summary")
st.write("Patients used:",len(df))
st.write("Deaths:",(y==0).sum())
st.write("Survivals:",(y==1).sum())

st.subheader("Prediction")

age = st.number_input("Age",0,120,30)
burn = st.slider("Burn %",0.0,100.0,30.0)

input_data = pd.DataFrame([[age,burn]],columns=["Age","Burn"])

prob_survival = model.predict_proba(input_data)[0][1]
prob_death = 1-prob_survival

st.write("Mortality risk:",round(prob_death*100,2),"%")
st.write("Survival probability:",round(prob_survival*100,2),"%")

auc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
st.write("Model AUC:",round(auc,3))
