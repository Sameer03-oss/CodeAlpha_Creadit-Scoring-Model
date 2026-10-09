# Credit Scoring Machine Learning Project

## 1. Project Overview

This project uses Machine Learning to predict whether a loan applicant is likely to have good or bad credit risk.

It trains and compares three Machine Learning models:

* Logistic Regression
* Decision Tree
* Random Forest

The project compares their accuracy and uses the best-performing model to make a prediction.

**Note:** This beginner project uses synthetic sample data for learning purposes. It does not use real customer data and should not be used to make actual lending decisions.

## 2. Technologies Used

* Python
* Pandas
* Scikit-learn
* Machine Learning

## 3. Project Structure

```text
Credit-Scoring-Project/
│
├── credit_scoring.py
└── README.md
```

## 4. Requirements

Install Python 3.10 or newer.

Install the required libraries using this command:

```bash
python -m pip install pandas scikit-learn
```

## 5. How to Run the Project

**Step 1:** Open the project folder in VS Code.

**Step 2:** Open the terminal.

**Step 3:** Install the required libraries.

```bash
python -m pip install pandas scikit-learn
```

**Step 4:** Run the Python file.

```bash
python credit_scoring.py
```

## 6. How It Works

1. Generates sample credit-related data.
2. Splits the data into training and testing sets.
3. Trains three Machine Learning models.
4. Calculates and compares model accuracy.
5. Selects the best-performing model.
6. Predicts the credit risk for a new sample applicant.

## 7. Expected Output

When you run the program, it displays the accuracy of the three models and a credit-risk prediction for a sample applicant.

The exact accuracy values may vary depending on the generated data and model settings.

## 8. Learning Objectives

* Understand basic Machine Learning concepts.
* Learn how to train and test models.
* Compare different classification algorithms.
* Make predictions using a trained model.
* Understand the basic idea of credit risk assessment.

## 9. Future Improvements

* Use a real credit dataset, such as the UCI Default of Credit Card Clients dataset.
* Add data visualization.
* Build a simple web interface using Streamlit.
* Evaluate models using precision, recall, and F1-score.
* Improve the model's reliability and fairness.

## 10. Author

**Project:** Credit Scoring Using Machine Learning
**Language:** Python
**Level:** Beginner
**Purpose:** Educational and academic use
