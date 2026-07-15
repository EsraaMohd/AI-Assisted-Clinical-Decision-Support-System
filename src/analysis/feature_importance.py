from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""
Feature Importance Analysis

This module visualizes the importance of features learned by a
Random Forest model.
"""

def plot_feature_importance(
    model,
    X: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    """
    Plot and save the top feature importances.

    Args:
        model:
            Trained Random Forest model.

        X:
            Feature dataframe.

        save_path:
            Output image path.

    Returns:
        DataFrame containing all features sorted by importance.
    """

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    print("\nTop 10 Important Features\n")
    print(importance_df.head(10))

    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title("Top 10 Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

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

    return importance_df