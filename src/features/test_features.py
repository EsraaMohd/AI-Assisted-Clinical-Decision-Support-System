from pathlib import Path

from src.features.split_features import split_features_target
from src.data.load_data import load_dataset
from src.features.build_features import encode_age, encode_binary_features, encode_categorical_features,build_features


def main():

    dataset_path = Path("data/raw/hospital_readmissions.csv")

    df = load_dataset(dataset_path)

    encoded_df = encode_age(df)

    print(encoded_df["age"].head(10))
    print(encoded_df["age"].unique())

    encoded_df = encode_binary_features(encoded_df)

    print(encoded_df[["change", "diabetes_med", "readmitted"]].head())
    
    # encoded_df = encode_age(df)
    # encoded_df = encode_binary_features(encoded_df)
    # encoded_df = encode_categorical_features(encoded_df)



    # print(encoded_df.shape)
    # print(encoded_df.columns.tolist())
    # print(encoded_df.head())


    processed_df = build_features(df)

    print(processed_df.shape)

    print(processed_df.head())

    processed_df = build_features(df)

    X, y = split_features_target(processed_df)

    print(X.shape)
    print(y.shape)

    print(type(X))
    print(type(y))

if __name__ == "__main__":
    main()