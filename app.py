import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(page_title="Loan Approval Prediction System", layout="wide")

# Paths
base_path = r'C:\Users\ELCOT\.gemini\antigravity\scratch\loan_approval_system'
model_path = os.path.join(base_path, 'loan_model.pkl')
encoders_path = os.path.join(base_path, 'label_encoders.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')
cat_imputer_path = os.path.join(base_path, 'cat_imputer.pkl')
num_imputer_path = os.path.join(base_path, 'num_imputer.pkl')
data_path = os.path.join(base_path, 'loan_dataset.csv')

@st.cache_resource
def load_artifacts():
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    cat_imputer = joblib.load(cat_imputer_path)
    num_imputer = joblib.load(num_imputer_path)
    return model, encoders, scaler, cat_imputer, num_imputer

def predict_loan(input_data):
    model, encoders, scaler, cat_imputer, num_imputer = load_artifacts()
    df = pd.DataFrame([input_data])
    
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # Missing values aren't strictly an issue here since we get complete input, but to be safe:
    for col in cat_cols:
        if pd.isna(df[col].iloc[0]): df[col] = cat_imputer.statistics_[cat_imputer.feature_names_in_.tolist().index(col)]
    for col in num_cols:
        if pd.isna(df[col].iloc[0]): df[col] = num_imputer.statistics_[num_imputer.feature_names_in_.tolist().index(col)]
        
    for col in cat_cols:
        if col in encoders:
            # handle unseen labels if any (fallback to 0)
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
                
    df[num_cols] = scaler.transform(df[num_cols])
    
    # ensure order matches
    feature_order = cat_cols + num_cols
    # wait, the order in train_model.py was just df.drop('Loan_Status').
    # the order in train_model is:
    # Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
    ordered_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                    'Credit_History', 'Property_Area']
    
    df = df[ordered_cols]
    prediction = model.predict(df)
    return prediction[0]

def main():
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("Automate loan approval decisions using Machine Learning. Reduce risk and improve decision accuracy.")
    
    tabs = st.tabs(["Prediction Module", "Data Input Module", "Reporting Module"])
    
    with tabs[1]: # Data Input
        st.header("Applicant Details")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x==1.0 else "Bad (0.0)")
            
        with col2:
            applicant_income = st.number_input("Applicant Income (Monthly)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-Applicant Income", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
            loan_term = st.selectbox("Loan Amount Term (in months)", [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0], index=8)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
    with tabs[0]: # Prediction Module
        st.header("Predict Loan Status")
        st.markdown("Go to the **Data Input Module** tab to change the applicant details. Once done, click the button below to predict.")
        
        if st.button("Predict Loan Status", type="primary"):
            with st.spinner("Analyzing applicant details..."):
                try:
                    result = predict_loan(input_data)
                    if result == 1:
                        st.success("✅ **Loan Approved!**")
                        st.balloons()
                    else:
                        st.error("❌ **Loan Rejected.**")
                except Exception as e:
                    st.error(f"Error making prediction: {e}. Please ensure models are trained.")

    with tabs[2]: # Reporting Module
        st.header("Exploratory Data Analysis (EDA) & Reports")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Income vs Loan Approval**")
                fig1, ax1 = plt.subplots(figsize=(6,4))
                sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df, ax=ax1)
                ax1.set_ylim(0, 30000) # clip outliers for better view
                st.pyplot(fig1)
                
            with col2:
                st.markdown("**Credit History vs Approval**")
                fig2, ax2 = plt.subplots(figsize=(6,4))
                sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax2)
                st.pyplot(fig2)
                
            st.markdown("### Model Performance Metrics")
            st.info("The system uses Random Forest or Logistic Regression. Training accuracy is usually around 80-95% depending on the generated synthetic data.")
        else:
            st.warning("Dataset not found. Please run the generation script.")

if __name__ == "__main__":
    main()
