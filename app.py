from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_credit_default_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_default_dataset.csv")

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

TARGET = "DEFAULT"
DROP_COLUMNS = ["CUST_ID", "DEFAULT", "R_HOUSING"]
FEATURES = [c for c in data.columns if c not in DROP_COLUMNS]

NUMERIC_COLUMNS = [c for c in FEATURES if pd.api.types.is_numeric_dtype(data[c])]
CATEGORICAL_COLUMNS = [c for c in FEATURES if c not in NUMERIC_COLUMNS]

DEFAULT_VALUES = {}
for col in NUMERIC_COLUMNS:
    DEFAULT_VALUES[col] = float(data[col].median())
for col in CATEGORICAL_COLUMNS:
    DEFAULT_VALUES[col] = str(data[col].mode().iloc[0])

# These fields include both text categories and 0/1 numeric indicators.
CATEGORY_OPTIONS = {
    "CAT_GAMBLING": ["No", "Low", "High"],
    "CAT_DEBT": ["0", "1"],
    "CAT_CREDIT_CARD": ["0", "1"],
    "CAT_MORTGAGE": ["0", "1"],
    "CAT_SAVINGS_ACCOUNT": ["0", "1"],
    "CAT_DEPENDENTS": ["0", "1"],
}

USER_FIELDS = [
    ("INCOME", "Annual Income", "number"),
    ("SAVINGS", "Savings", "number"),
    ("DEBT", "Total Debt", "number"),
    ("CREDIT_SCORE", "Credit Score", "number"),
    ("T_CLOTHING_12", "Clothing Spend / Year", "number"),
    ("T_EDUCATION_12", "Education Spend / Year", "number"),
    ("T_ENTERTAINMENT_12", "Entertainment Spend / Year", "number"),
    ("T_GROCERIES_12", "Groceries Spend / Year", "number"),
    ("T_HEALTH_12", "Health Spend / Year", "number"),
    ("T_HOUSING_12", "Housing Spend / Year", "number"),
    ("T_TAX_12", "Tax Spend / Year", "number"),
    ("T_TRAVEL_12", "Travel Spend / Year", "number"),
    ("T_UTILITIES_12", "Utilities Spend / Year", "number"),
]

USER_CATEGORIES = ["CAT_GAMBLING", "CAT_DEBT", "CAT_CREDIT_CARD",
                   "CAT_MORTGAGE", "CAT_SAVINGS_ACCOUNT", "CAT_DEPENDENTS"]

CATEGORY_LABELS = {
    "CAT_GAMBLING": "Gambling Category",
    "CAT_DEBT": "Debt Category",
    "CAT_CREDIT_CARD": "Credit Card",
    "CAT_MORTGAGE": "Mortgage",
    "CAT_SAVINGS_ACCOUNT": "Savings Account",
    "CAT_DEPENDENTS": "Dependents",
}

def build_features(form):
    row = DEFAULT_VALUES.copy()

    for col, _, _ in USER_FIELDS:
        value = form.get(col, "").strip()
        if value:
            try:
                row[col] = float(value)
            except ValueError:
                pass

    for col in USER_CATEGORIES:
        value = form.get(col, "").strip()
        if not value:
            raise ValueError(f"Please select a value for {CATEGORY_LABELS[col]}.")
        # Keep 0/1 categories numeric because that is how they are stored in the dataset.
        row[col] = float(value) if pd.api.types.is_numeric_dtype(data[col]) else value

    # Recreate the important engineered ratios from the dataset structure.
    income = max(float(row["INCOME"]), 1.0)
    savings = max(float(row["SAVINGS"]), 0.000001)
    debt = max(float(row["DEBT"]), 0.000001)

    row["R_SAVINGS_INCOME"] = savings / income
    row["R_DEBT_INCOME"] = debt / income
    row["R_DEBT_SAVINGS"] = debt / savings

    spend_pairs = [
        ("CLOTHING", "T_CLOTHING_12"),
        ("EDUCATION", "T_EDUCATION_12"),
        ("ENTERTAINMENT", "T_ENTERTAINMENT_12"),
        ("GROCERIES", "T_GROCERIES_12"),
        ("HEALTH", "T_HEALTH_12"),
        ("HOUSING", "T_HOUSING_12"),
        ("TAX", "T_TAX_12"),
        ("TRAVEL", "T_TRAVEL_12"),
        ("UTILITIES", "T_UTILITIES_12"),
    ]

    total_exp = 0.0
    for name, annual_col in spend_pairs:
        annual = float(row[annual_col])
        six_month = annual / 2.0
        row[f"T_{name}_6"] = six_month
        row[f"R_{name}"] = annual / max(income, 1.0)
        row[f"R_{name}_INCOME"] = annual / max(income, 1.0)
        row[f"R_{name}_SAVINGS"] = annual / savings
        row[f"R_{name}_DEBT"] = annual / debt
        total_exp += annual

    row["T_EXPENDITURE_12"] = total_exp
    row["T_EXPENDITURE_6"] = total_exp / 2.0
    row["R_EXPENDITURE"] = total_exp / max(income, 1.0)
    row["R_EXPENDITURE_INCOME"] = total_exp / max(income, 1.0)
    row["R_EXPENDITURE_SAVINGS"] = total_exp / savings
    row["R_EXPENDITURE_DEBT"] = total_exp / debt

    # R_HOUSING was removed during training because it is constant.
    return pd.DataFrame([row], columns=FEATURES)

def calculate_score(probability):
    # Project-specific score, not an official bureau/FICO score.
    return int(round(850 - probability * 550))

def risk_from_probability(probability):
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"

def explain_result(row, probability):
    income = float(row["INCOME"].iloc[0])
    savings = float(row["SAVINGS"].iloc[0])
    debt = float(row["DEBT"].iloc[0])
    credit = float(row["CREDIT_SCORE"].iloc[0])

    debt_ratio = debt / max(income, 1.0)
    savings_ratio = savings / max(income, 1.0)

    factors = []

    if debt_ratio > 0.5:
        factors.append(("Debt-to-income", "High", "Debt is high compared with annual income."))
    elif debt_ratio < 0.25:
        factors.append(("Debt-to-income", "Positive", "Debt is relatively low compared with annual income."))
    else:
        factors.append(("Debt-to-income", "Moderate", "Debt is at a moderate level compared with annual income."))

    if savings_ratio >= 0.25:
        factors.append(("Savings-to-income", "Positive", "Savings provide a stronger financial buffer relative to income."))
    elif savings_ratio < 0.10:
        factors.append(("Savings-to-income", "Watch", "Savings are relatively low compared with annual income."))
    else:
        factors.append(("Savings-to-income", "Moderate", "Savings provide a moderate financial buffer."))

    if credit >= 700:
        factors.append(("Credit score", "Positive", "The supplied credit score is relatively strong."))
    elif credit < 600:
        factors.append(("Credit score", "Watch", "The supplied credit score is relatively low."))
    else:
        factors.append(("Credit score", "Moderate", "The supplied credit score is in a middle range."))

    if probability >= 0.60:
        explanation = "The model estimates a higher likelihood of default. Review debt, savings and credit-related factors before making a lending decision."
    elif probability >= 0.30:
        explanation = "The model estimates a moderate likelihood of default. Some financial factors should be reviewed before a lending decision."
    else:
        explanation = "The model estimates a lower likelihood of default based on the supplied financial profile."

    return factors, explanation

def predict_row(row):
    probability = float(model.predict_proba(row)[0][1])
    prediction = int(model.predict(row)[0])
    risk = risk_from_probability(probability)
    score = calculate_score(probability)
    factors, explanation = explain_result(row, probability)
    return prediction, probability, risk, score, factors, explanation

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            for col, label, _ in USER_FIELDS:
                raw = request.form.get(col, "").strip()
                if not raw:
                    raise ValueError(f"Please enter {label}.")
                value = float(raw)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{label} must be a valid non-negative number.")
                if col == "INCOME" and value <= 0:
                    raise ValueError("Annual income must be greater than zero.")
                if col == "CREDIT_SCORE" and not 300 <= value <= 850:
                    raise ValueError("Credit score must be between 300 and 850.")
            for col in USER_CATEGORIES:
                if request.form.get(col, "").strip() not in CATEGORY_OPTIONS[col]:
                    raise ValueError(f"Please choose a valid option for {CATEGORY_LABELS[col]}.")
            row = build_features(request.form)
            prediction, probability, risk, score, factors, explanation = predict_row(row)
            result = {
                "prediction": "Default Risk Detected" if prediction == 1 else "Lower Default Risk",
                "probability": round(probability * 100, 2),
                "risk": risk,
                "score": score,
                "factors": factors,
                "explanation": explanation,
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "The model could not process these details. Check each field and try again."

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        user_fields=USER_FIELDS,
        user_categories=USER_CATEGORIES,
        category_labels=CATEGORY_LABELS,
        category_options=CATEGORY_OPTIONS,
    )

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("financial_file")
    if not file or not file.filename:
        return render_template(
            "index.html",
            result=None,
            error="Please choose a CSV financial file to upload.",
            form_values={},
            user_fields=USER_FIELDS,
            user_categories=USER_CATEGORIES,
            category_labels=CATEGORY_LABELS,
            category_options=CATEGORY_OPTIONS,
        )

    try:
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported for automatic import.")

        uploaded = pd.read_csv(file)
        available = [c for c in USER_FIELDS if c[0] in uploaded.columns]
        message = (
            f"Uploaded CSV contains {len(uploaded)} row(s). "
            f"{len(available)} supported financial fields were detected. "
            "For safety, review the values in the main form before prediction."
        )
        return render_template(
            "index.html",
            result=None,
            error=None,
            upload_message=message,
            form_values={},
            user_fields=USER_FIELDS,
            user_categories=USER_CATEGORIES,
            category_labels=CATEGORY_LABELS,
            category_options=CATEGORY_OPTIONS,
        )
    except Exception:
        return render_template(
            "index.html",
            result=None,
            error="CSV could not be read. Please upload a valid CSV file.",
            form_values={},
            user_fields=USER_FIELDS,
            user_categories=USER_CATEGORIES,
            category_labels=CATEGORY_LABELS,
            category_options=CATEGORY_OPTIONS,
        )

if __name__ == "__main__":
    app.run(debug=False)
