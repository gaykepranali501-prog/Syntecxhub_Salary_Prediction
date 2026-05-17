# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Dataset
data = pd.read_csv("dataset/Salary Data.csv")

# Display First 5 Rows
print(data.head())

# Features and Target
X = data[['Experience Years']]
y = data['Salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

# Example Prediction
experience = [[5]]
predicted_salary = model.predict(experience)

print("Predicted Salary for 5 Years Experience:", predicted_salary[0])

# Save Model
joblib.dump(model, "model/salary_model.pkl")

print("Model Saved Successfully!")
