from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert age groups into ordinal numerical values.
    """

    encoded_df = df.copy()   ## work on a copy of the original DataFrame to avoid modifying it directly

    age_mapping = {
        "[40-50)": 0,
        "[50-60)": 1,
        "[60-70)": 2,
        "[70-80)": 3,
        "[80-90)": 4,
        "[90-100)": 5,
    }

    encoded_df["age"] = encoded_df["age"].map(age_mapping)

    return encoded_df

def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary categorical features into numerical values.
    """

    encoded_df = df.copy()

    binary_mapping = {
        "no": 0,
        "yes": 1,
    }

    binary_columns = [
        "change",
        "diabetes_med",
        "readmitted",
    ]

    for column in binary_columns:
        encoded_df[column] = encoded_df[column].map(binary_mapping)

    return encoded_df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode nominal categorical features using One-Hot Encoding.
    """

    encoded_df = df.copy()

    categorical_columns = [
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "glucose_test",
        "A1Ctest",
    ]

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoded_array = encoder.fit_transform(
        encoded_df[categorical_columns]
    )

    encoded_features = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=encoded_df.index,
    )

    encoded_df = encoded_df.drop(columns=categorical_columns)

    encoded_df = pd.concat(
        [encoded_df, encoded_features],
        axis=1,
    )

    return encoded_df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline.

    Args:
        df: Raw input dataframe.

    Returns:
        Processed dataframe ready for machine learning.
    """

    processed_df = encode_age(df)

    processed_df = encode_binary_features(processed_df)

    processed_df = encode_categorical_features(processed_df)

    return processed_df