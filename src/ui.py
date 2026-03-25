import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("🏦 Credit Risk Analysis System")
st.markdown("Input borrower data to predict loan default probability.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_length = st.number_input(
        "Employment Length (Years)", min_value=0.0, value=2.0
    )
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )

with col2:
    st.subheader("Loan Details")
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=5000)
    loan_int_rate = st.number_input(
        "Interest Rate (%)", min_value=0.0, max_value=100.0, value=11.0
    )

with col3:
    st.subheader("Credit History")
    cb_person_default_on_file = st.selectbox("Historical Default?", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input(
        "Credit Hist Length (Years)", min_value=0, value=3
    )

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
    st.info(f"Loan-to-Income Ratio: {loan_percent_income:.2f}")

input_payload = {
    "person_age": int(person_age),
    "person_income": int(person_income),
    "person_home_ownership": person_home_ownership,
    "person_emp_length": float(person_emp_length),
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": int(loan_amnt),
    "loan_int_rate": float(loan_int_rate),
    "loan_percent_income": float(loan_percent_income),
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
}

if st.button("Predict Risk", use_container_width=True):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_payload)

        if response.status_code == 200:
            res = response.json()
            prob = res["probability"]
            pred = (
                "DEFAULT (High Risk)"
                if res["prediction"] == 1
                else "NON-DEFAULT (Low Risk)"
            )

            if res["prediction"] == 1:
                st.error(f"Prediction: {pred}")
            else:
                st.success(f"Prediction: {pred}")

            st.metric("Default Probability", f"{prob:.2%}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Could not connect to API. Is FastAPI running? Error: {e}")
