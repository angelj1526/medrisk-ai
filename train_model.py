import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("patient_data.csv")


# Input features
X = data[
    [
        "age",
        "blood_pressure",
        "glucose",
        "bmi"
    ]
]


# Target
y = data["risk"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create model
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("--------------------------------")
print("MedRisk AI Model Training")
print("--------------------------------")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("--------------------------------")


# Save model
with open(
    "patient_risk_model.pkl",
    "wb"
) as file:

    pickle.dump(model, file)


print("Model saved successfully!")
print("File: patient_risk_model.pkl")
