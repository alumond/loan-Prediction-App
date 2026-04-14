import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("loan_pipeline.pkl")

st.title("Loan Approval Prediction (Professional ML App)")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.number_input("Loan Term", min_value=0.0)
Credit_History = st.selectbox("Credit History", [0.0, 1.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Loan Approved ✅ (Confidence: {probability:.2f})")
    else:
        st.error(f"Loan Not Approved ❌ (Confidence: {probability:.2f})")
