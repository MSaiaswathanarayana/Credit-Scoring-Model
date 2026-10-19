# CodeAlpha Credit Scoring Model — Web Application

A machine learning web application that analyzes financial information and predicts credit default risk using a trained classification model.

## Project Overview

The Credit Scoring Model is developed as part of the CodeAlpha internship project. It uses historical financial data to estimate the likelihood of credit default and presents the prediction through a simple web interface.

The project demonstrates data preprocessing, feature engineering, model training, model evaluation, and deployment using Flask.

## Features

- Accepts financial information through a web form.
- Processes supported financial features using the same preprocessing approach used during training.
- Predicts credit default risk using a trained Logistic Regression model.
- Displays the model's predicted default probability and a risk category.
- Provides a plain-language explanation of the prediction.
- Highlights important financial factors used in the analysis.
- Supports structured CSV uploads for supported financial records.
- Validates user inputs and reports unsupported or invalid values.

*Note: Available features depend on the implementation and supported input fields in the application.*

## Technologies Used

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- HTML
- CSS
- JavaScript
- Jupyter Notebook

## Machine Learning Approach

The project uses a supervised machine learning classification approach.

The workflow includes:

1. Loading and inspecting the financial dataset.
2. Cleaning and preprocessing the data.
3. Preparing the input features and target variable.
4. Training classification models.
5. Comparing model performance using suitable evaluation metrics.
6. Selecting Logistic Regression based on the evaluation performed during development.
7. Integrating the trained model into the Flask web application.
8. Generating predictions for supported user inputs.

The project also compares Logistic Regression with Decision Tree and Random Forest models, where implemented in the training workflow.

## Dataset and Target Variable

The dataset contains financial attributes used to train a credit default classification model. The target column is named `DEFAULT`, and the dataset also contains a `CREDIT_SCORE` field.

The target column used by the model determines what the prediction represents. Please refer to the training notebook and dataset documentation for the exact target definition, feature list, and preprocessing steps.

**Important:** The application provides a project-specific machine learning prediction. It does not calculate an official credit bureau score.

## Prediction Results

Depending on the supported inputs and application implementation, the website can display:

- Predicted default probability
- Low, Medium, or High risk category
- Project-specific credit risk score
- Plain-language prediction explanation
- Key financial factors associated with the result

The risk category is an application-defined interpretation of the model output, not a bank's lending decision.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies
- pip
- Visual Studio Code or another Python development environment

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-scoring-model
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of this repository.

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start the Application

```bash
python app.py
```

If the application starts successfully, open the following address in your browser:

http://127.0.0.1:5000

If the trained model files are not included in the repository, follow the training instructions provided with the project before starting the application.

## CSV Upload

The application supports structured CSV files when their columns match the format expected by the model.

Before uploading a CSV file:

1. Review the supported feature names.
2. Use the same column names and compatible data types expected by the application.
3. Include the required financial fields.
4. Remove unrelated columns if they are not supported.
5. Check the displayed prediction results and validation messages.

A successful upload should process the supported records and generate predictions according to the implemented application workflow.

## Model Evaluation

Model performance should be assessed using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, when applicable

Refer to the training notebook or evaluation report for the measured results. Performance values should be reported only when they have been calculated using the project's actual evaluation data.

## Project Structure

The repository may contain the following files and folders, depending on the final implementation:

```text
credit-scoring-model/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── templates/
├── static/
├── models/
├── data/
└── notebooks/
```

- `app.py` — Flask application and prediction routes.
- `train_model.py` — model training script, if included.
- `requirements.txt` — Python dependencies.
- `templates/` — HTML templates.
- `static/` — CSS, JavaScript, and other frontend assets.
- `models/` — saved trained models and preprocessing objects, if required.
- `data/` — dataset files, subject to licensing and privacy restrictions.
- `notebooks/` — model development and evaluation notebooks, if included.

The actual repository structure may differ from this example.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Results may be unreliable for financial profiles outside the training data.
- Model probabilities are estimates, not guarantees of future repayment or default.
- The application is an educational machine learning project and is not a substitute for professional credit assessment.
- The displayed score is not an official CIBIL, Experian, Equifax, bank, or regulatory score.
- CSV uploads must follow the feature schema supported by the application.
- The project does not automatically extract financial information from bank-statement images or PDFs unless a separate document extraction feature has been implemented.

## Future Improvements

- Add more representative financial datasets.
- Improve model evaluation and validation.
- Add feature importance and explainability visualizations.
- Improve CSV validation and error handling.
- Develop a more detailed model monitoring and reporting workflow.
- Explore secure document extraction for supported financial documents.

## Internship Project

**Organization:** CodeAlpha  
**Project:** Credit Scoring Model  
**Domain:** Machine Learning  
**Application:** Flask Web Application

## Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions should not be used as the sole basis for real financial, lending, or credit approval decisions.

## Author

Developed as part of a machine learning internship project.
