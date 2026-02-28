import numpy as np
from sklearn.ensemble import IsolationForest

def compute_residuals(y_true,y_pred):
    return np.abs(y_true - y_pred)

def detect_anomalies(residuals,contamination=0.02):
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(residuals.reshape(-1,1))

    preds = np.where(preds == -1,1,0)
    return preds, iso