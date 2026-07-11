from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Args:
        file_path: Path to the dataset CSV file.

    Returns:
        A pandas DataFrame.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If the file extension is not CSV
            or the dataset contains no rows.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset '{file_path}' does not exist."
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, but got '{file_path.suffix}'."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Dataset '{file_path}' contains no records."
        )

    return df