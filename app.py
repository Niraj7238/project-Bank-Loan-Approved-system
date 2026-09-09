import os

# 1. Environment variables set for headless execution
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# 2. Imports
import joblib
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Loan Approval Predictor", page_icon="🏦", layout="wide"
)

st.title("🏦 Loan Approval Prediction Web App")
st.write(
    "Enter the applicant details on the left sidebar to check if the loan will"
    " be approved."
)


# Load saved model.pkl file
@st.cache_resource
def load_model():
  return joblib.load('model.pkl')


model = load_model()

# User Input Form (Sidebar)
st.sidebar.header("📋 Applicant Form")

credit_score = st.sidebar.number_input(
    "Credit Score", min_value=300, max_value=850, value=700
)
dti_ratio = st.sidebar.number_input(
    "DTI Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01
)
applicant_income = st.sidebar.number_input(
    "Applicant Income ($)", min_value=0, value=8000, step=500
)
coapplicant_income = st.sidebar.number_input(
    "Co-applicant Income ($)", min_value=0, value=2000, step=500
)
loan_amount = st.sidebar.number_input(
    "Loan Amount ($)", min_value=0, value=15000, step=1000
)
loan_term = st.sidebar.selectbox(
    "Loan Term (Months)", options=[12, 24, 36, 60, 120, 360], index=3
)
age = st.sidebar.slider("Age", 18, 100, 35)
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
existing_loans = st.sidebar.selectbox("Existing Loans", [0, 1, 2, 3, 4, 5])
savings = st.sidebar.number_input("Savings ($)", min_value=0, value=10000, step=1000)
collateral_value = st.sidebar.number_input(
    "Collateral Value ($)", min_value=0, value=25000, step=1000
)

employment_status = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Self-employed", "Unemployed"]
)
marital_status = st.sidebar.selectbox(
    "Marital Status", ["Single", "Married", "Divorced"]
)
loan_purpose = st.sidebar.selectbox(
    "Loan Purpose", ["Personal", "Car", "Business", "Education", "Home"]
)
property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)
education_level = st.sidebar.selectbox(
    "Education Level", ["Graduate", "Not Graduate"]
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
employer_category = st.sidebar.selectbox(
    "Employer Category", ["Private", "Government", "MNC", "Business"]
)

# Convert inputs into DataFrame
input_data = pd.DataFrame([{
    "Applicant_Income": applicant_income,
    "Coapplicant_Income": coapplicant_income,
    "Employment_Status": employment_status,
    "Age": age,
    "Marital_Status": marital_status,
    "Dependents": dependents,
    "Credit_Score": credit_score,
    "Existing_Loans": existing_loans,
    "DTI_Ratio": dti_ratio,
    "Savings": savings,
    "Collateral_Value": collateral_value,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term,
    "Loan_Purpose": loan_purpose,
    "Property_Area": property_area,
    "Education_Level": education_level,
    "Gender": gender,
    "Employer_Category": employer_category,
}])

# App Body Layout
col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("🎯 Approval Prediction")
  if st.button("Predict Loan Status", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
      st.success(
          f"✅ **Loan Approved!**\n\nProbability: `{probability*100:.1f}%`"
      )
    else:
      st.error(
          f"❌ **Loan Rejected.**\n\nProbability: `{probability*100:.1f}%`"
      )

with col2:
  st.subheader("📊 Model Status")
  st.info("Model successfully loaded from `model.pkl`.")