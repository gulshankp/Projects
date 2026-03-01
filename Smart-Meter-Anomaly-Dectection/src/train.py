import numpy as np
import joblib
import os
from data_preprocessing import load_data, preprocess
from lstm_model import build_lstm
from anomaly_detection import compute_residuals, detect_anomalies

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
df = load_data("data/smart_meter_data.csv")

# -----------------------------
# Preprocess
# -----------------------------
X, y, scaler, df_original = preprocess(df)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Build & Train Model
# -----------------------------
model = build_lstm((X_train.shape[1], 1))

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# Predict
# -----------------------------
y_pred = model.predict(X_test).flatten()

# -----------------------------
# Residuals
# -----------------------------
residuals = compute_residuals(y_test, y_pred)

# -----------------------------
# Detect anomalies
# -----------------------------
pred_labels, iso_model = detect_anomalies(residuals)

# -----------------------------
# TRUE anomaly labels
# -----------------------------
# Align anomaly labels with test set
true_labels = df_original["anomaly"].values[-len(y_test):]

np.save("results/true_labels.npy", true_labels.astype(int))
np.save("results/predicted_labels.npy", pred_labels.astype(int))

# -----------------------------
# Save timestamps and actual values
# -----------------------------
timestamps = df_original["timestamp"].values[-len(y_test):]
actual_values = df_original["consumption"].values[-len(y_test):]

np.save("results/timestamps.npy", timestamps)
np.save("results/actual_values.npy", actual_values)

# -----------------------------
# Save models
# -----------------------------
model.save("models/lstm_model.h5")
joblib.dump(iso_model, "models/isolation_forest.pkl")

print("Training completed successfully!")