from pathlib import Path
import pandas as pd

from src.data.load_data import load_dataset


def summarize_categorical_columns(df: pd.DataFrame) -> None:
    """
    Print summary information for all categorical columns.
    """

    categorical_columns = df.select_dtypes(include="object").columns

    for column in categorical_columns:

        print("=" * 60)
        print(f"Column: {column}")
        print(f"Unique values: {df[column].nunique()}")

        print("\nTop values:")

        print(df[column].value_counts().head(10))

        print()

if __name__ == "__main__":

    dataset_path = Path("data/raw/hospital_readmissions.csv")

    df = load_dataset(dataset_path)

    summarize_categorical_columns(df)
