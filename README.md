# CodeAlpha Credit Scoring Model

## About the Project

I developed this project as part of my internship at CodeAlpha to understand how Machine Learning can be used to analyze financial information and estimate credit risk.

The application takes financial details such as annual income, savings, debt, expenses, and other financial information. It uses a trained Machine Learning model to estimate the possibility of loan default and displays the result through a simple web interface.

My main goal was to build a project that connects Machine Learning with a website so users can enter their financial details and understand the prediction in an easy way.

## Features

- Enter financial details manually through the website.
- Upload a CSV file containing supported financial information.
- Estimate the probability of default using a trained Machine Learning model.
- Display a project-specific credit score and risk category.
- Provide a simple explanation of the prediction and important financial factors.
- Use a web interface built with Python and Flask.

## Technologies Used

- **Python** – Main programming language
- **Pandas and NumPy** – Data handling and processing
- **Scikit-learn** – Machine Learning model training and evaluation
- **Logistic Regression** – Selected prediction model
- **Flask** – Connecting the Machine Learning model to the website
- **HTML and CSS** – Website structure and design
- **Jupyter Notebook** – Data analysis, model training, and evaluation

## Machine Learning Approach

I worked with structured financial data containing information such as income, savings, debt, expenditure patterns, financial ratios, and credit score.

I explored Logistic Regression, Decision Tree, and Random Forest models and compared their performance using evaluation metrics, including precision, recall, F1-score, and ROC-AUC.

Logistic Regression was selected based on the F1-score comparison conducted during the project.

The trained model is then used by the Flask application to process financial information and generate a prediction.

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

Open the downloaded folder in Visual Studio Code.

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open the website

Open the following URL in your browser:

`http://127.0.0.1:5000`

You can then enter financial details manually or upload a supported CSV file.

## Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## What I Learned

While working on this project, I learned about data preprocessing, feature engineering, training and evaluating Machine Learning models, and integrating a trained model with a Flask web application.

I also gained practical experience in handling user inputs, working with CSV files, and presenting prediction results through a website.

## Important Note

This project is developed for educational and internship purposes. The displayed credit score is a project-specific estimate and is not an official CIBIL, FICO, Experian, Equifax, bank, or regulatory credit score.

The prediction should not be used as the sole basis for real loan approvals or financial decisions.

## Acknowledgement

I would like to thank CodeAlpha for providing me with the opportunity to work on this project and improve my practical understanding of Machine Learning and web application development.
