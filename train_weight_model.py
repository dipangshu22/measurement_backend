import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Dummy dataset
heights = np.array([150, 160, 170, 180, 190]).reshape(-1, 1)
weights = np.array([50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(heights, weights)

joblib.dump(model, "weight_model.pkl")

print("Model trained and saved!")