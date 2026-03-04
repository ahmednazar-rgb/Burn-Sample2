import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = [
[100,18,0],[35,44,0],[90,60,0],[30,31,1],[75,18,0],[13,6,1],[60,4,0],[20,36,1],
[10,5,1],[8,0.75,1],[20,29,1],[100,32,0],[15,1.5,1],[15,17,1],[10,4,1],[15,25,1],
[100,15,0],[100,35,0],[35,2.5,1],[30,30,1],[50,33,1],[45,2,1],[70,29,0],[60,40,0],
[50,45,0],[20,27,1],[35,1.5,1],[95,18,0],[50,25,0],[15,20,1],[10,40,1],[18,39,1],
[20,27,1],[60,31,1],[25,26,1],[40,26,1],[15,20,1],[30,25,1],[80,16,0],[20,30,1],
[30,30,1],[10,30,1],[25,1,1],[65,25,0],[100,26,0],[35,37,0],[5,2,1],[70,32,0],
[10,5,1],[40,29,1],[80,32,0],[100,28,0],[40,24,1],[50,37,0],[20,1.5,1],[35,38,1],
[5,11,1],[100,18,0],[90,45,0],[20,21,1],[25,28,1],[40,22,1],[60,3,0]
]

df = pd.DataFrame(data, columns=["Burn","Age","Outcome"])

X = df[["Burn","Age"]]
y = df["Outcome"]

preprocess = ColumnTransformer([
("num", StandardScaler(), ["Burn","Age"])
])

model = Pipeline([
("prep", preprocess),
("clf", LogisticRegression(max_iter=2000))
])

model.fit(X,y)

st.title("Burn Mortality Predictor")

age = st.number_input("Age (years)",0.0,120.0)
burn = st.slider("Burn Percentage",0,100)

if st.button("Predict"):
    input_df = pd.DataFrame([[burn,age]],columns=["Burn","Age"])
    prob_survive = model.predict_proba(input_df)[0][1]
    prob_die = 1 - prob_survive

    st.write("Mortality Risk:", round(prob_die*100,2), "%")
    st.write("Survival Probability:", round(prob_survive*100,2), "%")
