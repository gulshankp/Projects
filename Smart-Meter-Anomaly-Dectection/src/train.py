import numpy as np
import joblib
from data_preprocessing import load_data, preprocess
from lstm_model import build_lstm
from anomaly_detection import compute_residuals, detect_anomalies

df = load_data("data/smart_meter_data.csv")

# Preprocess
X, y, scaler, df_original = preprocess(df)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = build_lstm((X_train.shape[1],1))

#train
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# Predict
y_pred = model.predict(X_test)

#Residuals
residuals = compute_residuals(y_test,y_pred.flatten())

# Detect anomalies
pred_labels, iso_model = detect_anomalies(residuals)

#save models
model.save("models/lstm_model.h5")
joblib.dump(iso_model,"models/isolation_forest.pkl")

print("Training completed successfully!")