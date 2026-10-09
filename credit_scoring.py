
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Create a sample dataset
# Replace this with a real credit dataset for your final project.
X, y = make_classification(
    n_samples=1000,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    random_state=42
)

features = [
    "Income", "Loan_Amount", "Credit_History",
    "Payment_Delay", "Debt", "Loan_Term"
]

X = pd.DataFrame(X, columns=features)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 4. Train and compare models
best_model = None
best_accuracy = 0

print("CREDIT SCORING MODEL RESULTS")
print("-" * 35)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2%} accuracy")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# 5. Show the best model
print("\nBest model:", type(best_model).__name__)
print(f"Test accuracy: {best_accuracy:.2%}")

# 6. Predict for a new example
new_customer = pd.DataFrame(
    [[0.5, -0.2, 1.0, 0.3, -0.4, 0.8]],
    columns=features
)

prediction = best_model.predict(new_customer)[0]

print("\nNEW CUSTOMER PREDICTION")

if prediction == 1:
    print("Higher estimated risk class")
else:
    print("Lower estimated risk class")

print("\nNote: This uses synthetic data for demonstration.")
print("It is not suitable for real lending decisions.")