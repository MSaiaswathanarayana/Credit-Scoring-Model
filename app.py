from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_credit_default_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_default_dataset.csv")

# Saved model (Logistic Regression pipeline from the notebook).
# Used for full-format CSVs, which contain every feature it was trained on.
model = joblib.load(MODEL_PATH)

data = pd.read_csv(DATA_PATH)

TARGET = "DEFAULT"
DROP_COLUMNS = ["CUST_ID", "DEFAULT", "R_HOUSING"]
FEATURES = [c for c in data.columns if c not in DROP_COLUMNS]

NUMERIC_COLUMNS = [c for c in FEATURES if pd.api.types.is_numeric_dtype(data[c])]
CATEGORICAL_COLUMNS = [c for c in FEATURES if c not in NUMERIC_COLUMNS]

CATEGORY_OPTIONS = {
    "CAT_GAMBLING": ["No", "Low", "High"],
    "CAT_DEBT": ["0", "1"],
    "CAT_CREDIT_CARD": ["0", "1"],
    "CAT_MORTGAGE": ["0", "1"],
    "CAT_SAVINGS_ACCOUNT": ["0", "1"],
    "CAT_DEPENDENTS": ["0", "1"],
}
ZERO_ONE_CATEGORIES = ["CAT_DEBT", "CAT_CREDIT_CARD", "CAT_MORTGAGE",
                       "CAT_SAVINGS_ACCOUNT", "CAT_DEPENDENTS"]
YES_NO_ALIASES = {"yes": "1", "y": "1", "true": "1",
                  "no": "0", "n": "0", "false": "0"}

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
NUMBER_LABELS = {col: label for col, label, _ in USER_FIELDS}

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

SPEND_PAIRS = [
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

MAX_TABLE_ROWS = 300


# ---------------------------------------------------------------------------
# Feature engineering for the simplified inputs (form + simplified CSV)
# ---------------------------------------------------------------------------

def simplified_feature_frame(df, total_expense=None):
    """Derive model features from the fields a user can enter.

    Works on a whole DataFrame at once. Training and prediction both use this
    function, so the companion model sees identical feature definitions.
    total_expense: optional Series of user-supplied annual totals.
    """
    income = df["INCOME"].clip(lower=1.0)
    savings = df["SAVINGS"].clip(lower=1e-6)
    debt = df["DEBT"].clip(lower=1e-6)

    out = pd.DataFrame(index=df.index)
    out["INCOME"] = df["INCOME"].astype(float)
    out["SAVINGS"] = df["SAVINGS"].astype(float)
    out["DEBT"] = df["DEBT"].astype(float)
    out["CREDIT_SCORE"] = df["CREDIT_SCORE"].astype(float)
    out["R_SAVINGS_INCOME"] = savings / income
    out["R_DEBT_INCOME"] = debt / income
    out["R_DEBT_SAVINGS"] = debt / savings

    total = pd.Series(0.0, index=df.index)
    for name, col in SPEND_PAIRS:
        annual = df[col].astype(float)
        out[col] = annual
        out[f"R_{name}_INCOME"] = annual / income
        out[f"R_{name}_SAVINGS"] = annual / savings
        out[f"R_{name}_DEBT"] = annual / debt
        total = total + annual

    if total_expense is not None:
        total = total_expense.astype(float)

    out["T_EXPENDITURE_12"] = total
    out["R_EXPENDITURE_INCOME"] = total / income
    out["R_EXPENDITURE_SAVINGS"] = total / savings
    out["R_EXPENDITURE_DEBT"] = total / debt

    for col in ZERO_ONE_CATEGORIES:
        out[col] = df[col].astype(float)
    out["CAT_GAMBLING_LOW"] = (df["CAT_GAMBLING"].astype(str) == "Low").astype(float)
    out["CAT_GAMBLING_HIGH"] = (df["CAT_GAMBLING"].astype(str) == "High").astype(float)

    return out


def train_simplified_model():
    """Train the companion model on the dataset, using only user-enterable fields."""
    X = simplified_feature_frame(data)
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
    ])
    estimator.fit(X_train, y_train)

    probabilities = estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, probabilities >= 0.5) * 100, 1),
        "f1": round(f1_score(y_test, probabilities >= 0.5), 3),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 3),
        "test_rows": int(len(y_test)),
    }
    return estimator, metrics


simple_model, SIMPLE_METRICS = train_simplified_model()


def default_class_index(estimator):
    for i, label in enumerate(estimator.classes_):
        if str(label) == "1":
            return i
    raise RuntimeError("The model does not contain the default class labelled 1.")


SAVED_DEFAULT_INDEX = default_class_index(model)
SIMPLE_DEFAULT_INDEX = default_class_index(simple_model)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def is_missing(raw):
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip() == ""
    try:
        return bool(pd.isna(raw))
    except (TypeError, ValueError):
        return False


def parse_number(raw, label, column=None):
    if is_missing(raw):
        raise ValueError(f"Please enter {label}.")

    text = str(raw).strip().replace(",", "")  # accepts 1,00,000 and 100,000
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.")

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a valid non-negative number.")
    if column == "INCOME" and value <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if column == "CREDIT_SCORE" and not 300 <= value <= 850:
        raise ValueError("Credit score must be between 300 and 850.")
    return value


def normalize_category(column, raw):
    label = CATEGORY_LABELS[column]
    if is_missing(raw):
        raise ValueError(f"Please select a value for {label}.")

    text = str(raw).strip()
    try:
        number = float(text)
        if number.is_integer():
            text = str(int(number))  # "1", "1.0" and 1 all become "1"
    except ValueError:
        pass

    options = CATEGORY_OPTIONS[column]
    if text in options:
        return text

    lowered = text.lower()
    if column in ZERO_ONE_CATEGORIES and lowered in YES_NO_ALIASES:
        return YES_NO_ALIASES[lowered]
    for option in options:
        if option.lower() == lowered:
            return option

    raise ValueError(
        f"Invalid value for {label}: '{text}'. "
        f"Allowed values: {', '.join(options)}."
    )


def simplified_rows_from_mapping_list(records, total_column=None):
    """Validate a list of simplified input mappings into a DataFrame."""
    clean = []
    for position, record in enumerate(records):
        line = position + 2
        try:
            row = {}
            for col, label, _ in USER_FIELDS:
                row[col] = parse_number(record.get(col), label, column=col)
            for col in USER_CATEGORIES:
                row[col] = normalize_category(col, record.get(col))
            clean.append(row)
        except ValueError as exc:
            raise ValueError(f"Row {line}: {exc}" if len(records) > 1 else str(exc))
    return pd.DataFrame(clean)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_score(probability):
    # Project-specific score (300-850 range), not an official bureau/FICO score.
    return int(round(850 - probability * 550))


def risk_from_probability(probability):
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"


def explain_result(row, probability):
    income = float(row["INCOME"])
    savings = float(row["SAVINGS"])
    debt = float(row["DEBT"])
    credit = float(row["CREDIT_SCORE"])

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


def score_frame(frame, estimator, default_index, raw_rows):
    """Score every row.

    frame: model-ready features for `estimator`.
    raw_rows: DataFrame with INCOME, SAVINGS, DEBT, CREDIT_SCORE for explanations.
    """
    probabilities = estimator.predict_proba(frame)[:, default_index]
    predictions = estimator.predict(frame)

    results = []
    for i in range(len(frame)):
        probability = float(probabilities[i])
        factors, explanation = explain_result(raw_rows.iloc[i], probability)
        results.append({
            "prediction": "Default Risk Detected" if int(predictions[i]) == 1 else "Lower Default Risk",
            "probability": round(probability * 100, 2),
            "risk": risk_from_probability(probability),
            "score": calculate_score(probability),
            "factors": factors,
            "explanation": explanation,
        })
    return results


def score_simplified(rows, total_expense=None):
    """Score simplified inputs (rows = DataFrame of validated values)."""
    frame = simplified_feature_frame(rows, total_expense)
    return score_frame(frame[simple_model.feature_names_in_], simple_model,
                       SIMPLE_DEFAULT_INDEX, rows)


def score_full_features(df):
    """Score full-format CSV rows with the saved model (exact)."""
    frame = build_full_feature_frame(df)
    return score_frame(frame, model, SAVED_DEFAULT_INDEX, frame)


def build_full_feature_frame(df):
    missing_rows = df[FEATURES].isna().any(axis=1)
    if missing_rows.any():
        first = missing_rows[missing_rows].index[0]
        raise ValueError(f"CSV row {first + 2} contains missing values.")

    frame = df[FEATURES].copy().replace([np.inf, -np.inf], np.nan)

    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    bad = frame[NUMERIC_COLUMNS].isna().any(axis=1)
    if bad.any():
        first = bad[bad].index[0]
        raise ValueError(f"CSV row {first + 2} has a non-numeric value.")
    for col in NUMERIC_COLUMNS:
        frame[col] = frame[col].astype(float)

    for col in CATEGORICAL_COLUMNS:
        frame[col] = frame[col].astype(str).str.strip()

    return frame.reset_index(drop=True)


def render_page(**context):
    page = dict(
        result=None,
        error=None,
        upload_message=None,
        csv_results=[],
        csv_summary=None,
        csv_method=None,
        csv_has_actual=False,
        csv_truncated=0,
        form_values={},
        user_fields=USER_FIELDS,
        user_categories=USER_CATEGORIES,
        category_labels=CATEGORY_LABELS,
        category_options=CATEGORY_OPTIONS,
        simple_metrics=SIMPLE_METRICS,
    )
    page.update(context)
    return render_template("index.html", **page)


@app.errorhandler(413)
def file_too_large(_):
    return render_page(error="The uploaded file is larger than 20 MB."), 413


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            rows = simplified_rows_from_mapping_list([request.form])
            total = parse_number(
                request.form.get("major_annual_expenses"),
                "Major annual expenses",
            )
            total_series = pd.Series([total], index=rows.index)
            result = score_simplified(rows, total_series)[0]
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Manual prediction failed")
            error = "The model could not process these details. Check each field and try again."

    return render_page(result=result, error=error, form_values=form_values)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("financial_file")

    if not file or not file.filename:
        return render_page(error="Please select a CSV file.")

    try:
        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a CSV file.")

        uploaded = pd.read_csv(file, encoding="utf-8-sig")
        uploaded.columns = [str(c).strip() for c in uploaded.columns]

        if uploaded.empty:
            raise ValueError("The uploaded CSV has no customer records.")

        full_mode = all(column in uploaded.columns for column in FEATURES)

        if full_mode:
            scores = score_full_features(uploaded)
            method = "Exact (saved model, full model features)"
        else:
            required = [col for col, _, _ in USER_FIELDS] + USER_CATEGORIES
            missing = [col for col in required if col not in uploaded.columns]
            if missing:
                raise ValueError(
                    "Missing CSV columns: " + ", ".join(missing) +
                    ". Upload either the full model-feature CSV or a CSV "
                    "with the 13 simplified fields plus the 6 category columns."
                )

            records = uploaded.to_dict(orient="records")
            rows = simplified_rows_from_mapping_list(records)

            total_series = None
            total_column = next(
                (c for c in ("major_annual_expenses", "T_EXPENDITURE_12") if c in uploaded.columns),
                None,
            )
            if total_column:
                totals = []
                for position, value in enumerate(uploaded[total_column].tolist()):
                    if is_missing(value):
                        totals.append(np.nan)
                        continue
                    try:
                        totals.append(parse_number(value, "Major annual expenses"))
                    except ValueError as exc:
                        raise ValueError(f"Row {position + 2}: {exc}")
                total_series = pd.Series(totals, index=rows.index)
                # Use the user's total where given; otherwise the sum of categories.
                fallback = simplified_feature_frame(rows)["T_EXPENDITURE_12"]
                total_series = total_series.fillna(fallback)

            scores = score_simplified(rows, total_series)
            method = "Estimated (simplified-field model)"

        customer_column = next(
            (c for c in ("CUST_ID", "customer_id", "Customer_ID") if c in uploaded.columns),
            None,
        )
        has_actual = TARGET in uploaded.columns

        csv_results = []
        comparable = []
        for position, item in enumerate(scores):
            source = uploaded.iloc[position]
            record = {
                "row_number": position + 1,
                "customer_id": str(source[customer_column]) if customer_column else "—",
                **item,
                "actual": "—",
            }
            if has_actual and not is_missing(source[TARGET]):
                actual_value = int(float(source[TARGET]))
                record["actual"] = "Default" if actual_value == 1 else "No default"
                comparable.append((item["prediction"] == "Default Risk Detected") == (actual_value == 1))
            csv_results.append(record)

        total = len(csv_results)
        accuracy = round(100 * sum(comparable) / len(comparable), 2) if comparable else None

        summary = {
            "total": total,
            "low": sum(1 for r in csv_results if r["risk"] == "Low Risk"),
            "medium": sum(1 for r in csv_results if r["risk"] == "Medium Risk"),
            "high": sum(1 for r in csv_results if r["risk"] == "High Risk"),
            "avg_score": round(sum(r["score"] for r in csv_results) / total),
            "defaults": sum(1 for r in csv_results if r["prediction"] == "Default Risk Detected"),
            "accuracy": accuracy,
        }

        return render_page(
            upload_message=f"Scored {total} customer record(s) from your CSV.",
            csv_results=csv_results[:MAX_TABLE_ROWS],
            csv_summary=summary,
            csv_method=method,
            csv_has_actual=has_actual,
            csv_truncated=max(0, total - MAX_TABLE_ROWS),
        )

    except ValueError as exc:
        return render_page(error=f"CSV prediction failed: {exc}")
    except Exception:
        app.logger.exception("CSV prediction failed")
        return render_page(
            error="CSV prediction failed. Check that the file is a valid CSV in the expected format."
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
