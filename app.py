import streamlit as st
import pandas as pd
import joblib

st.title("Prediksi Churn Pelanggan Telco")

model = joblib.load("model/churn_model.joblib")
df = pd.read_csv("data/telco_churn.csv").drop(columns=["customerID", "Churn"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

inputs = {}
for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "str":
        inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
    else:
        inputs[col] = st.number_input(col, value=float(df[col].median()))

if st.button("Prediksi"):
    data = pd.DataFrame([inputs])
    result = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]
    if result == 1:
        st.error(f"Pelanggan berpotensi CHURN (probabilitas {proba:.0%})")
    else:
        st.success(f"Pelanggan cenderung BERTAHAN (probabilitas churn {proba:.0%})")