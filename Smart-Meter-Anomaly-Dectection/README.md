⚡ Smart Meter Anomaly Detection

Hybrid LSTM + Isolation Forest Approach

📌 Overview

This project implements a hybrid anomaly detection pipeline for smart meter energy consumption data.

Instead of directly classifying anomalies, the system separates:

•	Temporal pattern learning (LSTM)
•	Deviation detection (Isolation Forest on residuals)
This design makes the detection more robust and realistic for real-world energy systems.

________________________________________

🧠 Methodology

Step 1 — LSTM Forecasting

The LSTM model learns normal time-series consumption patterns from historical data.

Step 2 — Residual Computation

Residual = |Actual − Predicted|

Large residuals indicate abnormal deviations.

Step 3 — Isolation Forest

Isolation Forest is applied on residual values to detect anomalies.
________________________________________

📊 Dataset

•	6 months of hourly smart meter data

•	Daily seasonality pattern

•	Gaussian noise added

•	40 injected anomalies (~1% of total data)

The dataset is highly imbalanced, making precision–recall analysis critical.

________________________________________

📈 Model Evaluation & Trade-Off Analysis

🔹 Model Version 1 — 1% Contamination (Conservative Detection)

•	Precision (Anomaly): 1.00

•	Recall (Anomaly): 0.82

•	F1-score: 0.90

•	Accuracy: 1.00

✔ No false positives

⚠ Missed 2 anomalies

Interpretation:

This configuration produces highly reliable alerts but may miss some anomalies.

Best suited for:

•	Cost-sensitive inspection systems

•	Scenarios where false alarms are expensive

🔹 Model Version 2 — 2% Contamination (Aggressive Detection)

•	Precision (Anomaly): 0.61

•	Recall (Anomaly): 1.00

•	F1-score: 0.76

•	Accuracy: 0.99

✔ Detected all anomalies

⚠ Some false positives

Interpretation:

This configuration prioritizes sensitivity and ensures no anomaly is missed.

Best suited for:

•	Safety-critical infrastructure

•	Power grid monitoring

•	Energy theft detection

________________________________________

🎯 Key Insight

Anomaly detection is not about maximizing accuracy.

Because the dataset is imbalanced (~1% anomalies), Precision and Recall are more meaningful metrics than Accuracy.

There is a clear trade-off:

•	Higher Recall → Fewer missed anomalies

•	Higher Precision → Fewer false alarms

Model configuration should align with business risk tolerance.

________________________________________

🛠 Tech Stack

•	Python

•	TensorFlow / Keras (LSTM)

•	Scikit-learn (Isolation Forest)

•	NumPy / Pandas

•	Matplotlib

________________________________________

🔮 Future Improvements

•	Real-world smart meter dataset integration

•	Hyperparameter optimization

•	ROC & Precision-Recall curve analysis

•	Threshold tuning strategies

•	Real-time streaming implementation

________________________________________

📌 Conclusion

This project demonstrates:

•	Hybrid time-series anomaly detection

•	Residual-based modeling strategy

•	Handling imbalanced datasets

•	Precision–Recall trade-off tuning

The results show how model behavior changes based on contamination level, reinforcing the importance of aligning ML systems with business objectives.

