import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import os

def train():
    data_path = r'C:\Users\ELCOT\.gemini\antigravity\scratch\loan_approval_system\loan_dataset.csv'
    df = pd.read_csv(data_path)
    
    # 1. Preprocessing
    # Drop Loan_ID
    df = df.drop('Loan_ID', axis=1)
    
    # Identify cat and num columns
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # Imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')
    
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Encoding
    le_dict = {}
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    # Map target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Scaling
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Model Training
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    print("Model Evaluation:\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}:")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("-" * 30)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy {best_acc:.4f}")
    
    # 3. Save Model and Artifacts
    output_dir = r'C:\Users\ELCOT\.gemini\antigravity\scratch\loan_approval_system'
    joblib.dump(best_model, os.path.join(output_dir, 'loan_model.pkl'))
    joblib.dump(le_dict, os.path.join(output_dir, 'label_encoders.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(cat_imputer, os.path.join(output_dir, 'cat_imputer.pkl'))
    joblib.dump(num_imputer, os.path.join(output_dir, 'num_imputer.pkl'))
    print("Saved model and preprocessing objects.")

if __name__ == "__main__":
    train()
