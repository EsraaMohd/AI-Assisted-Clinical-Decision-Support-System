from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_calibration_curve(
    y_true,
    y_prob,
    save_path,
):

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
    )

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=2,
        label="XGBoost",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Perfect Calibration",
    )

    plt.xlabel("Mean Predicted Probability")

    plt.ylabel("Observed Frequency")

    plt.title(
        f"Calibration Curve\nBrier Score = {brier:.4f}"
    )

    plt.grid(True)

    plt.legend()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("\nCalibration curve saved.")

    print(f"Brier Score: {brier:.4f}")