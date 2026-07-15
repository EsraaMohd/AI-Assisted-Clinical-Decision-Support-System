from pathlib import Path

import joblib

"""
Model Persistence Module

This module saves trained machine learning models.
"""


def save_model(
    model,
    file_path: Path,
) -> None:
    """
    Save a trained model to disk.

    Args:
        model:
            Trained machine learning model.

        file_path:
            Destination path.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        file_path,
    )

    print(f"\nModel saved successfully:\n{file_path}")