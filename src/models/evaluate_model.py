import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
"""
Model Evaluation Module

This module evaluates classification models.
"""

def evaluate_model(
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Evaluate classification model performance.

    Args:
        y_true:
            Ground truth labels.

        y_pred:
            Predicted labels.

        y_prob:
            Predicted probabilities for the positive class.
            Used only for ROC-AUC.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
    )

    recall = recall_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
    )

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix\n")

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print(cm)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    display.plot()

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()

    if y_prob is not None:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        print(f"\nROC-AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob,
        )

        plt.figure(figsize=(6, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.3f}",
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "--",
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.show()