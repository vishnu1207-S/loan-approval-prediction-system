import pandas as pd
import numpy as np

def generate_loan_data(num_records=1000):
    np.random.seed(42)
    
    data = {
        'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, num_records + 1)],
        'Gender': np.random.choice(['Male', 'Female'], num_records, p=[0.75, 0.25]),
        'Married': np.random.choice(['Yes', 'No'], num_records, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], num_records, p=[0.55, 0.17, 0.18, 0.10]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], num_records, p=[0.78, 0.22]),
        'Self_Employed': np.random.choice(['No', 'Yes'], num_records, p=[0.85, 0.15]),
        'ApplicantIncome': np.random.randint(150, 80000, num_records),
        'CoapplicantIncome': np.random.randint(0, 40000, num_records),
        'LoanAmount': np.random.randint(9, 700, num_records),
        'Loan_Amount_Term': np.random.choice([12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0], num_records),
        'Credit_History': np.random.choice([1.0, 0.0], num_records, p=[0.84, 0.16]),
        'Property_Area': np.random.choice(['Semiurban', 'Urban', 'Rural'], num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce a bit of logic so it's not totally random
    # Approval is more likely if Credit_History == 1, ApplicantIncome + CoapplicantIncome > 5000, LoanAmount < 500
    
    loan_status = []
    for _, row in df.iterrows():
        score = 0
        if row['Credit_History'] == 1.0:
            score += 5
        if row['ApplicantIncome'] + row['CoapplicantIncome'] > 6000:
            score += 2
        if row['LoanAmount'] < 300:
            score += 1
        if row['Education'] == 'Graduate':
            score += 1
            
        if score >= 6:
            loan_status.append('Y')
        else:
            loan_status.append('N')
            
    df['Loan_Status'] = loan_status
    
    # Randomly add some nulls to simulate real world data
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        mask = np.random.rand(num_records) < 0.05
        df.loc[mask, col] = np.nan
        
    df.to_csv(r'C:\Users\ELCOT\.gemini\antigravity\scratch\loan_approval_system\loan_dataset.csv', index=False)
    print(f'Generated loan_dataset.csv with {num_records} records.')

if __name__ == "__main__":
    generate_loan_data()
