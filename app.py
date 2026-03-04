import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

data = [
[100,18,"F",0],
[35,44,"F",0],
[90,60,"F",0],
[30,31,"M",1],
[75,18,"M",0],
[13,6,"F",1],
[60,4,"M",0],
[20,36,"F",1],
[10,5,"M",1],
[8,0.75,"M",1],
[20,29,"F",1],
[100,32,"M",0],
[15,1.5,"M",1],
[15,17,"M",1],
[10,4,"M",1],
[15,25,"F",1],
[100,15,"F",0],
[100,35,"F",0],
[35,2.5,"M",1],
[30,30,"M",1],
[50,33,"M",1],
[45,2,"M",1],
[70,29,"F",0],
[60,40,"F",0],
[50,45,"F",0],
[20,27,"M",1],
[35,1.5,"M",1],
[95,18,"F",0],
[50,25,"F",0],
[15,20,"F",1],
[10,40,"M",1],
[18,39,"M",1],
[20,27,"M",1],
[60,31,"M",1],
[25,26,"M",1],
[40,26,"M",1],
[15,20,"F",1],
[30,25,"F",1],
[80,16,"F",0],
[20,30,"M",1],
[30,30,"F",1],
[10,30,"M",1],
[25,1,"M",1],
[65,25,"F",0],
[100,26,"F",0],
[35,37,"M",0],
[5,2,"M",1],
[70,32,"F",0],
[10,5,"M",1],
[40,29,"M",1],
[80,32,"F",0],
[100,28,"F",0],
[40,24,"M",1],
[50,37,"F",0],
[20,1.5,"F",1],
[35,38,"F",1],
[5,11,"M",1],
[100,18,"M",0],
[90,45,"M",0],
[20,21,"M",1],
[25,28,"M",1],
[40,22,"M",1],
[60,3,"M",0]
]

df = pd.DataFrame(data, columns=["Burn","Age","Sex","Outcome"])

X = df[["Burn","Age","Sex"]]
y = df["Outcome"]

preprocess = ColumnTransformer([
("num", StandardScaler(), ["Burn","Age"]),
("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex"])
])

model = Pipeline([
("prep", preprocess),
("clf", LogisticRegression(max_iter=2000))
])

model.fit(X,y)

st.title("Burn Mortality Predictor")

age = st.number_input("Age",0.0,100.0)
burn = st.slider("Burn Percentage",0,100)
sex = st.selectbox("Sex",["M","F"])

if st.button("Predict"):
    input_df = pd.DataFrame([[burn,age,sex]],columns=["Burn","Age","Sex"])
    prob_survive = model.predict_proba(input_df)[0][1]
    prob_die = 1 - prob_survive

    st.write("Mortality Risk:",round(prob_die*100,2),"%")
    st.write("Survival Probability:",round(prob_survive*100,2),"%")
