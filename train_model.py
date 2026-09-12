import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("patient_data.csv")

# Input features
X = data[["age", "blood_pressure", "glucose", "bmi"]]

# Target
y = data["risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
with open("patient_risk_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")