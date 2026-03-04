import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

# -----------------------------
# Data from "Burn sample 2.pdf"
# -----------------------------
rows = [
    # Name, BurnPct, Age, Sex, DOA, DOD, Outcome (0=death, 1=survive)
    ("Safa adnan",100,18,"F","4/7/2018","12/7/2018",0),
    ("Amina mahmod",35,44,"F","5/7/2018","10/7/2018",0),
    ("Nora ali",90,60,"F","9/7/2018","11/7/2018",0),
    ("Ayman yohana",30,31,"M","10/7/2018","21/7/2018",1),
    ("Abdo abdulbast",75,18,"M","10/7/2018","18/7/2018",0),
    ("Aynor azad",13,6,"F","10/7/2018","15/7/2018",1),
    ("Shorash hasan",60,4,"M","13/7/2018","17/7/2018",0),
    ("Buhar sadeeq",20,36,"F","15/7/2018","28/7/2018",1),
    ("Barzan mohsen",10,5,"M","15/7/2018","28/7/2018",1),
    ("Hiva hishiar",8,0.75,"M","19/7/2018","28/7/2018",1),   # 9 months
    ("Xonaf mohsen",20,29,"F","22/7/2018","28/7/2018",1),
    ("Ibrahim ahmed",100,32,"M","25/7/2018","26/7/2018",0),
    ("Ismael ato",15,1.5,"M","27/7/2018","2/8/2018",1),
    ("Hawraz adanan",15,17,"M","27/7/2018","9/8/2018",1),
    ("Khadija mohamed",10,4,"M","29/7/2018","15/8/2018",1),
    ("Nishtiman ali",15,25,"F","31/7/2018","16/8/2018",1),
    ("Zaynab ziad",100,15,"F","1/8/2018","5/8/2018",0),

    ("Zina ahmed",100,35,"F","3/8/2018","3/8/2018",0),
    ("Rebaz renas",35,2.5,"M","7/8/2018","28/8/2018",1),
    ("Amir ramadan",30,30,"M","8/8/2018","26/8/2018",1),
    ("Yasr ahmed",50,33,"M","9/8/2018","1/9/2018",1),
    ("Ali salih",45,2,"M","13/8/2018","13/9/2018",1),
    ("Ziman mohamed",70,29,"F","15/8/2018","26/8/2018",0),
    ("Haihat jawhar",60,40,"F","17/8/2018","30/8/2018",0),
    ("Lale mohamed",50,45,"F","20/8/2018","3/9/2018",0),
    ("Imad rajab",20,27,"M","21/8/2018","26/8/2018",1),
    ("Osama fars",35,1.5,"M","22/8/2018","15/9/2018",1),
    ("Randa badal",95,18,"F","28/8/2018","29/8/2018",0),
    ("Amira ismael",50,25,"F","29/8/2018","11/9/2018",0),
    ("Drxshan xalid",15,20,"F","30/8/2018","8/9/2018",1),
    ("Xisro marwan",10,40,"M","1/9/2018","3/9/2018",1),
    ("Karwan tahr",18,39,"M","3/9/2018","8/9/2018",1),
    ("Amin mohammed",20,27,"M","11/9/2018","18/9/2018",1),
    ("Qanaa sulaiman",60,31,"M","11/9/2018","23/10/2018",1),
    ("Atala jasm",25,26,"M","12/9/2018","22/9/2018",1),
    ("Jamil mustafa",40,26,"M","14/9/2018","17/9/2018",1),

    ("Manal salm",15,20,"F","14/9/2018","17/9/2018",1),
    ("Sania jamil",30,25,"F","15/9/2018","11/10/2018",1),
    ("Obaida ali",80,16,"F","15/9/2018","22/9/2018",0),
    ("Fadi fawaz",20,30,"M","16/9/2018","22/9/2018",1),
    ("Sozan sharif",30,30,"F","18/9/2018","23/9/2018",1),
    ("Mohammed omar",10,30,"M","20/9/2018","22/9/2018",1),
    ("Gashbin majeed",25,1,"M","23/9/2018","4/10/2018",1),
    ("Laila mirza",65,25,"F","24/9/2018","4/10/2018",0),
    ("Nora ibrahim",100,26,"F","1/10/2018","1/10/2018",0),
    ("Abdulsatar al ola",35,37,"M","1/10/2018","4/10/2018",0),
    ("Wisam rizgar",5,2,"M","1/10/2018","7/10/2018",1),
    ("Fatima majeed",70,32,"F","5/10/2018","5/10/2018",0),
    ("Bewar abdal",10,5,"M","6/10/2018","8/10/2018",1),
    ("Hogr salih",40,29,"M","7/10/2018","24/10/2018",1),
    ("Jihan tahr",80,32,"F","8/10/2018","13/10/2018",0),
    ("Asma malak",100,28,"F","12/10/2018","13/10/2018",0),
    ("Sardar mohamed",40,24,"M","13/10/2018","27/10/2018",1),
    ("Tahrir mahmoud",50,37,"F","14/10/2018","15/10/2018",0),
    ("Rozheen mohamed",20,1.5,"F","19/10/2018","29/10/2018",1),
    ("Evan mustafa",35,38,"F","21/10/2018","5/12/2018",1),
    ("Alind othman",5,11,"M","21/10/2018","22/10/2018",1),
    ("Ahmed mohammed",100,18,"M","24/10/2018","27/10/2018",0),
    ("Askandar xodeda",90,45,"M","26/10/2018","28/10/2018",0),
    ("Samir ahmed",20,21,"M","28/10/2018","5/11/2018",1),
    ("Moahmed amer",25,28,"M","28/10/2018","3/11/2018",1),
    ("Imad hussein",40,22,"M","29/10/2018","2/12/2018",1),
    ("Dildar dilgash",60,3,"M","29/10/2018","4/11/2018",0),
]

df = pd.DataFrame(rows, columns=["Name","BurnPct","Age","Sex","DOA","DOD","Outcome"])

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%d/%m/%Y")

df["DOA_dt"] = df["DOA"].apply(parse_date)
df["DOD_dt"] = df["DOD"].apply(parse_date)
df["LOS_days"] = (df["DOD_dt"] - df["DOA_dt"]).dt.days.clip(lower=0)

# -----------------------------
# Models
# -----------------------------
# Mortality model: uses Age + BurnPct + Sex
X_mort = df[["BurnPct","Age","Sex"]].copy()
y_mort = df["Outcome"].astype(int)

preprocess_mort = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["BurnPct","Age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
    ]
)

mort_model = Pipeline(
    steps=[
        ("prep", preprocess_mort),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)
mort_model.fit(X_mort, y_mort)

# LOS model: per your requirement uses ONLY Age + BurnPct
X_los = df[["BurnPct","Age"]].copy()
y_los = df["LOS_days"].astype(float)

preprocess_los = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["BurnPct","Age"]),
    ],
    remainder="drop"
)

los_model = Pipeline(
    steps=[
        ("prep", preprocess_los),
        ("reg", Ridge(alpha=1.0)),
    ]
)
los_model.fit(X_los, np.log1p(y_los))

def predict(age_years: float, burn_pct: float, sex: str):
    # Mortality
    x_m = pd.DataFrame([{"BurnPct": burn_pct, "Age": age_years, "Sex": sex}])
    p_survive = float(mort_model.predict_proba(x_m)[0, 1])
    p_die = 1.0 - p_survive

    # LOS
    x_l = pd.DataFrame([{"BurnPct": burn_pct, "Age": age_years}])
    log_los = float(los_model.predict(x_l)[0])
    los_days = float(np.expm1(log_los))
    los_days = max(0.0, los_days)

    return p_die, los_days

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Burn Outcome Predictor", layout="centered")
st.title("Burn Mortality + Expected Length of Stay")

st.caption("Trained only on your uploaded dataset (Burn sample 2). Small dataset → unstable estimates.")

age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=25.0, step=0.5)
burn = st.slider("Burn % (TBSA)", min_value=0, max_value=100, value=20)
sex = st.selectbox("Sex", ["M", "F"])

if st.button("Predict"):
    p_die, los_days = predict(age_years=float(age), burn_pct=float(burn), sex=str(sex))

    st.subheader("Results")
    st.metric("Predicted mortality risk", f"{p_die*100:.2f}%")
    st.metric("Expected length of stay", f"{los_days:.1f} days")

    st.caption("LOS model uses only Age + Burn% as requested. Mortality model uses Age + Burn% + Sex.")
