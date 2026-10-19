"""Credit default prediction helper.

Usage:
    python src/predict.py

The example below loads the saved model and predicts one row from the
provided dataset. Replace the example input with a new customer's data
using the same feature columns as the training dataset.
"""

from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_credit_default_model.pkl"
DATA_PATH = ROOT / "data" / "credit_default_dataset.csv"

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

# The notebook removes the identifier and constant feature before training.
input_data = data.drop(columns=["CUST_ID", "R_HOUSING", "DEFAULT"]).iloc[[0]]

prediction = int(model.predict(input_data)[0])
probability = float(model.predict_proba(input_data)[0, 1])

print("Credit Default Prediction")
print("-------------------------")
print("Predicted class:", prediction)
print("Probability of default:", round(probability, 4))

if prediction == 1:
    print("Result: HIGHER RISK / DEFAULT")
else:
    print("Result: LOWER RISK / NO DEFAULT")
