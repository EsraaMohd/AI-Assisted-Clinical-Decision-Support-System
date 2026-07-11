import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> tuple:
    """
    Return the dataset shape as (rows, columns).
    """
    return df.shape


def get_column_names(df: pd.DataFrame) -> list:
    """
    Return a list of all column names.
    """
    return df.columns.tolist()


def get_data_types(df: pd.DataFrame) -> dict:
    """
    Return the data type of each column.
    """
    return df.dtypes.astype(str).to_dict()


def get_missing_values(df: pd.DataFrame) -> dict:
    """
    Return the number of missing values in each column.
    """
    return df.isnull().sum().to_dict()


def get_duplicate_count(df: pd.DataFrame) -> int:
    """
    Return the number of duplicated rows.
    """
    return int(df.duplicated().sum())


def generate_summary(df: pd.DataFrame) -> dict:
    """
    Generate a complete summary of the dataset.
    """

    return {
        "shape": get_dataset_shape(df),
        "columns": get_column_names(df),
        "data_types": get_data_types(df),
        "missing_values": get_missing_values(df),
        "duplicate_rows": get_duplicate_count(df),
    }