# Loan Approval Prediction System

## Project Overview
The Loan Approval Prediction System is a Machine Learning project that predicts whether a loan application will be approved or rejected based on the applicant's information. This system helps financial institutions make faster and more consistent loan approval decisions.

## Features
- Data Preprocessing
- Loan Approval Prediction
- Machine Learning Model
- Streamlit Web Application
- User-Friendly Interface

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Files
- preprocessing.py
- training.py
- app.py
- loan_cleaned.csv
- loan_approval_model.pkl
- requirements.txt
- README.md

## Dataset
The dataset contains applicant information such as:
- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

The target variable is:
- Loan Status (Approved / Not Approved)

## How to Run

1. Install the required libraries:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python training.py
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Author
**Nazish Safdar**

BS Computer Science

Women University Mardan