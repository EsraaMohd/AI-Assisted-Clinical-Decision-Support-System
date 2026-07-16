from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def threshold_analysis(
    y_true,
    probabilities,
    save_plot_path,
    save_csv_path,
):

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({

            "Threshold": threshold,

            "Accuracy": accuracy_score(
                y_true,
                predictions,
            ),

            "Precision": precision_score(
                y_true,
                predictions,
            ),

            "Recall": recall_score(
                y_true,
                predictions,
            ),

            "F1": f1_score(
                y_true,
                predictions,
            ),

        })

    results = pd.DataFrame(results)

    print("\n")
    print(results.round(4))

    save_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        save_csv_path,
        index=False,
    )

    plt.figure(figsize=(10,6))

    plt.plot(
        results["Threshold"],
        results["Recall"],
        marker="o",
        linewidth=2,
        label="Recall",
    )

    plt.plot(
        results["Threshold"],
        results["Precision"],
        marker="o",
        linewidth=2,
        label="Precision",
    )

    plt.plot(
        results["Threshold"],
        results["F1"],
        marker="o",
        linewidth=2,
        label="F1-score",
    )

    plt.xlabel("Threshold")

    plt.ylabel("Score")

    plt.title(
        "Threshold Optimization"
    )

    plt.grid(True)

    plt.legend()

    save_plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("\nThreshold analysis saved successfully.")