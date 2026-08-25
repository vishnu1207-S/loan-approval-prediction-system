# Loan Approval Prediction System

A complete Machine Learning pipeline and interactive web application to predict loan approvals based on applicant details.

## Overview
This system automates loan approval decisions using Machine Learning. It predicts whether a loan application should be approved or rejected to reduce manual decision-making errors, speed up loan processing, and help assess risk efficiently.

## Features
- **Data Generation**: Generates synthetic loan application data for training.
- **Model Training**: Preprocesses data, handles missing values, and trains multiple classification models (Logistic Regression, Decision Tree, Random Forest) to find the best performing model.
- **Web UI**: An interactive Streamlit dashboard featuring:
  - **Data Input Module**: For submitting new loan applicant details.
  - **Prediction Module**: Instant Approved/Rejected classification.
  - **Reporting Module**: EDA visualizations (Income vs Approval, Credit History vs Approval) and model metrics.

## Tech Stack
- **Python 3**
- **Pandas & NumPy** (Data Handling)
- **Scikit-learn** (Machine Learning)
- **Matplotlib & Seaborn** (Data Visualization)
- **Streamlit** (Web Application Framework)

## Setup & Installation

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate Synthetic Data:**
   ```bash
   python generate_data.py
   ```
4. **Train the Model:**
   ```bash
   python train_model.py
   ```
5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage
Once the Streamlit app is running, navigate to `http://localhost:8501`. Switch between tabs to view the reports, input applicant data, and get instant loan approval predictions.
