from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.models.save_model import save_model
from src.models.evaluate_model import evaluate_model
from src.data.load_data import load_dataset
from src.features.build_features import build_features
from src.features.split_features import split_features_target
from pathlib import Path


DATA_PATH = Path("data/raw/hospital_readmissions.csv")

MODEL_PATH = Path("saved_models/logistic_regression.pkl")

def main():

    dataset_path = Path("data/raw/hospital_readmissions.csv")

    df = load_dataset(dataset_path)

    processed_df = build_features(df)

    X, y = split_features_target(processed_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42,
    stratify=y,)


    print(X_train.shape)
    print(X_test.shape)

    print(y_train.shape)
    print(y_test.shape)


    model = LogisticRegression(random_state=42, max_iter=1000,)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions,)

    print(f"Accuracy: {accuracy:.4f}")

    evaluate_model(y_test,predictions,)


    
    save_model(
        model=model,
        file_path=MODEL_PATH,
    )


if __name__ == "__main__":
    main()    