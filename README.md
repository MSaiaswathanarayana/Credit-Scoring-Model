# Credit Scoring Model – Web Application

## About the Project

This project is part of my CodeAlpha internship. I developed a Credit Scoring Model using machine learning to predict credit default risk based on financial information.

The main objective of this project is to understand how machine learning can be used to analyze financial data and estimate the possibility of a person defaulting on their credit obligations. I also developed a web application so that users can interact with the model easily without running the code manually.

## Project Objectives

- Analyze financial data and identify useful patterns.
- Train machine learning models to predict credit default risk.
- Compare different classification algorithms.
- Display predictions through a simple web interface.
- Understand how machine learning models can be integrated into real-world applications.

## Features

- Enter financial details through a web form.
- Predict credit default risk using a trained machine learning model.
- Display the predicted default probability and risk category.
- Provide a simple explanation of the prediction.
- Highlight important financial factors used in the analysis.
- Upload a supported CSV file containing financial records.
- Validate inputs and display error messages for invalid values.

## Technologies Used

- **Python** – Main programming language.
- **Flask** – Used to develop the web application.
- **pandas and NumPy** – Used for data handling and preprocessing.
- **scikit-learn** – Used for machine learning model development and evaluation.
- **HTML and CSS** – Used to create the website interface.
- **JavaScript** – Used for frontend interactions, where required.
- **Jupyter Notebook** – Used for model development and experimentation, if included.

## Machine Learning Algorithms

During this project, I worked with classification algorithms such as:

- Logistic Regression
- Decision Tree
- Random Forest

These algorithms were considered for predicting credit default risk. Logistic Regression was selected for the web application based on the model evaluation performed during development.

## How the Project Works

First, I collected and explored the financial dataset to understand its columns, values, and target variable. Then, I prepared the data for training by performing the required preprocessing steps.

Next, I trained and evaluated the machine learning models using suitable classification metrics. After selecting a model, I integrated it into the Flask web application.

Users can enter financial information through the website or upload a supported CSV file. The application processes the available inputs and displays the prediction, estimated default probability, risk category, and explanation according to the implemented model.

## How to Run the Project

### Requirements

- Python installed on your computer.
- Visual Studio Code or another Python development environment.
- The project files and required dependencies.

### Installation Steps

**Step 1: Open the project folder**

Download or clone this repository and open the extracted folder in Visual Studio Code.

**Step 2: Install the required libraries**

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

**Step 3: Run the application**

```bash
python app.py
```

**Step 4: Open the website**

Open your browser and visit:

http://127.0.0.1:5000

You can then enter financial information or upload a supported CSV file to test the application.

## Model Evaluation

The models should be evaluated using classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC where applicable.

These metrics help measure how well the model distinguishes between the target classes. The actual evaluation results should be taken from the project's training notebook or evaluation report.

## Project Limitations

This project is developed for educational purposes, and its predictions depend on the quality and characteristics of the training dataset.

The predicted probability and risk category are specific to this application. They are not official credit scores issued by CIBIL, Experian, Equifax, banks, or other financial institutions.

CSV uploads must follow the column format expected by the application. Automatic extraction of financial details from bank-statement images or PDF documents is not included unless implemented separately.

The prediction should not be used as the only basis for real lending or financial decisions.

## Future Improvements

In the future, the project could be improved by using larger and more representative datasets, adding better visualizations, improving prediction explanations, and strengthening CSV validation. Additional model evaluation and testing could also help identify areas where performance can be improved.

## What I Learned

While working on this project, I gained practical experience in data preprocessing, training machine learning models, comparing classification algorithms, and evaluating model performance. I also learned how to connect a trained machine learning model with a Flask web application and present its results through a user-friendly interface.

This project helped me understand the complete workflow of a basic machine learning application, from preparing the dataset to displaying predictions on a website.

## Internship Details

**Organization:** CodeAlpha

**Project Name:** Credit Scoring Model

**Domain:** Machine Learning

**Application Type:** Web Application

**Developed using:** Python and Flask
