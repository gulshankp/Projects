import numpy as np
import pandas as pd

np.random.seed(42)

def generate_smart_meter_data():
    hours = 24 * 180  # 6 months hourly data
    time = pd.date_range(start="2023-01-01", periods=hours, freq="h")

    # Base consumption pattern (daily seasonality)
    daily_pattern = 50 + 15 * np.sin(np.arange(hours) * (2 * np.pi / 24))

    # Add noise
    noise = np.random.normal(0, 3, hours)

    consumption = daily_pattern + noise

    # Inject anomalies
    anomaly_indices = np.random.choice(hours, size=40, replace=False)
    consumption[anomaly_indices] += np.random.uniform(30, 60, size=40)

    anomaly_label = np.zeros(hours)
    anomaly_label[anomaly_indices] = 1

    df = pd.DataFrame({
        "timestamp": time,
        "consumption": consumption,
        "anomaly": anomaly_label
    })

    df.to_csv("smart_meter_data.csv", index=False)
    print("Dataset generated successfully!")

# 👇 This must be at the top level, not indented inside the function
if __name__ == "__main__":
    generate_smart_meter_data()