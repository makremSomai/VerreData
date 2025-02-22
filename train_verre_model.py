import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load cleaned dataset
df = pd.read_csv("cleaned_verredata.csv")

# Define features (input) and targets (output)
X = df.iloc[:, 1:-2]  # Composition columns
y_conductivity = df["Conductivité thermique"]
y_resistance = df["Résistance Mécanique"]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train_c, y_test_c = train_test_split(X_scaled, y_conductivity, test_size=0.2, random_state=42)
X_train, X_test, y_train_r, y_test_r = train_test_split(X_scaled, y_resistance, test_size=0.2, random_state=42)

# Train models
model_conductivity = RandomForestRegressor(n_estimators=200, random_state=42)
model_resistance = RandomForestRegressor(n_estimators=200, random_state=42)

model_conductivity.fit(X_train, y_train_c)
model_resistance.fit(X_train, y_train_r)

# Evaluate models
mse_c = mean_squared_error(y_test_c, model_conductivity.predict(X_test))
mse_r = mean_squared_error(y_test_r, model_resistance.predict(X_test))

print(f"Conductivity Model MSE: {mse_c:.4f}")
print(f"Resistance Model MSE: {mse_r:.4f}")

# Save models
joblib.dump(model_conductivity, "model_conductivity_verre.pkl")
joblib.dump(model_resistance, "model_resistance_verre.pkl")
joblib.dump(scaler, "scaler_verre.pkl")
joblib.dump(X.columns.tolist(), "feature_names_verre.pkl")
print("✅ Models saved!")
