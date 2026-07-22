import streamlit as st
import joblib
import numpy as np

# ==========================
# Load Model & Scaler
# ==========================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

st.title("💰 Loan Approval Prediction System")
st.write("Enter the applicant details below.")

# ==========================
# Input Fields
# ==========================
age = st.number_input("Age", 18, 100, 25)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

education = st.selectbox(
    "Education",
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
)
education_map = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4
}
education = education_map[education]

income = st.number_input("Person Income", 0, 1000000, 50000)

experience = st.number_input("Employee Experience", 0, 50, 2)

home = st.selectbox(
    "Home Ownership",
    ["Rent", "Own", "Mortgage", "Other"]
)
home_map = {
    "Rent": 0,
    "Own": 1,
    "Mortgage": 2,
    "Other": 3
}
home = home_map[home]

loan_amount = st.number_input("Loan Amount", 1000, 1000000, 100000)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "Education",
        "Medical",
        "Personal",
        "Venture",
        "Home Improvement",
        "Debt Consolidation"
    ]
)
intent_map = {
    "Education": 0,
    "Medical": 1,
    "Personal": 2,
    "Venture": 3,
    "Home Improvement": 4,
    "Debt Consolidation": 5
}
loan_intent = intent_map[loan_intent]

interest = st.number_input("Loan Interest Rate", 0.0, 30.0, 10.0)

loan_percent = st.number_input("Loan Percentage", 0.0, 1.0, 0.20)

credit_history = st.number_input("Credit History", 0, 30, 5)

credit_score = st.number_input("Credit Score", 300, 850, 650)

previous_loan = st.selectbox("Previous Loan Default", ["No", "Yes"])
previous_loan = 0 if previous_loan == "No" else 1

# ==========================
# Prediction
# ==========================
if st.button("Predict Loan Status"):

    data = np.array([[
        age,
        gender,
        education,
        income,
        experience,
        home,
        loan_amount,
        loan_intent,
        interest,
        loan_percent,
        credit_history,
        credit_score,
        previous_loan
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")