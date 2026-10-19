# CodeAlpha Credit Scoring Model — Web Application

This is the deployment/interface version of the CodeAlpha Credit Scoring Model project.

## What this website does

1. Accepts a user's financial information.
2. Builds the same type of financial ratios used by the training dataset.
3. Loads the trained Logistic Regression model.
4. Predicts default risk.
5. Displays:
   - Default probability
   - Low / Medium / High risk
   - Project-specific credit score
   - Plain-language explanation
   - Main financial factors
6. Can check a structured CSV upload for supported financial fields.

## Important model note

The project dataset contains `DEFAULT` as the target and financial variables such as income, savings, debt, financial ratios, expenditure patterns, categorical financial information and `CREDIT_SCORE`.

The model was compared with Logistic Regression, Decision Tree and Random Forest. Logistic Regression was selected based on the strongest F1-score among the tested models.

The website's displayed score is a project-specific risk score. It is NOT an official FICO, CIBIL, Experian, Equifax, bank or regulatory score.

## Why there is no automatic bank-statement image/PDF reader

Financial documents can contain sensitive values and OCR can misread numbers. This simple project therefore accepts CSV data for structured upload and keeps the main financial form for controlled prediction. Do not claim that the application automatically reads bank statements unless a tested OCR/document extraction system is actually added.

## Run the project

Open this folder in VS Code.

```bash
pip install -r requirements.txt
python app.py
```

Then open:

`http://127.0.0.1:5000`


## Internship explanation

The website is the deployment layer. The notebook is the training/evaluation layer. Both belong to the same Credit Scoring Model project.
