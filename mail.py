# main.py
# Water Quality Check - Machine Learning Model
# Detects waterborne diseases from water quality parameters

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
# (Replace 'water_data.csv' with your dataset file)
data = pd.read_csv("water_data.csv")

# Step 2: Preprocess data
X = data.drop("label", axis=1)   # features
y = data["label"]                # target (e.g., "Safe" or "Unsafe")

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Function to predict new water sample
def predict_water_quality(sample):
    result = model.predict([sample])[0]
    return "Safe to Drink" if result == 1 else "Unsafe Water"

# Example: test with a sample (values depend on dataset columns)
# sample = [pH, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
sample = [7.0, 150, 20000, 8.0, 300, 500, 10, 60, 4.0]
print("Prediction:", predict_water_quality(sample))
