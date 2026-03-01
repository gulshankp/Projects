import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def evaluate_model(true_labels_path, predicted_labels_path):
    """
    Load saved labels and print classification report.
    """
    true_labels = np.load(true_labels_path)
    predicted = np.load(predicted_labels_path)

    print("True labels length:", len(true_labels))
    print("Predicted length:", len(predicted))
    print("\nClassification Report:\n")
    print(classification_report(true_labels, predicted))

    return true_labels, predicted


def plot_anomalies(timestamps, actual_values, anomaly_flags, save_path=None):
    """
    Plot actual consumption and highlight detected anomalies.
    """

    plt.figure(figsize=(15, 6))

    plt.plot(
        timestamps,
        actual_values,
        label="Actual Consumption"
    )

    plt.scatter(
        timestamps[anomaly_flags == 1],
        actual_values[anomaly_flags == 1],
        color="red",
        label="Detected Anomalies"
    )

    plt.title("Hybrid LSTM + Isolation Forest Anomaly Detection")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Evaluate classification performance
    true_labels, predicted = evaluate_model(
        "results/true_labels.npy",
        "results/predicted_labels.npy"
    )

    # Optional: load time-series for visualization

    try:
        timestamps = np.load("results/timestamps.npy", allow_pickle=True)
        actual_values = np.load("results/actual_values.npy")

        plot_anomalies(
            timestamps=timestamps,
            actual_values=actual_values,
            anomaly_flags=predicted,
            save_path="results/anomaly_detection_with_2%.png"
        )
    except FileNotFoundError:
        print("Time-series files not found. Skipping visualization.")