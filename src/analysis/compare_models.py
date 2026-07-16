import pandas as pd

from tabulate import tabulate


def compare_models():

    results = [

        {
            "Model": "Logistic Regression",
            "Accuracy": 0.6114,
            "Precision": 0.6309,
            "Recall": 0.4181,
            "F1": 0.5029,
            "ROC-AUC": None,
        },

        {
            "Model": "Random Forest",
            "Accuracy": 0.6022,
            "Precision": 0.5901,
            "Recall": 0.5040,
            "F1": 0.5437,
            "ROC-AUC": None,
        },

        {
            "Model": "Tuned Random Forest",
            "Accuracy": 0.6132,
            "Precision": 0.6045,
            "Recall": 0.5130,
            "F1": 0.5550,
            "ROC-AUC": 0.6427,
        },

        {
            "Model": "XGBoost",
            "Accuracy": 0.6030,
            "Precision": 0.5900,
            "Recall": 0.5130,
            "F1": 0.5500,
            "ROC-AUC": 0.6371,
        },
        {
            "Model": "Tuned XGBoost",
            "Accuracy": 0.5978,
            "Precision": 0.5801,
            "Recall": 0.5236,
            "F1": 0.5504,
            "ROC-AUC": 0.6318,
      },
    ]

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)

    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="github",
            showindex=False,
            floatfmt=".4f",
        )
    )

    print("\n")

    best = df.loc[
        df["Recall"].idxmax()
    ]

    print("=" * 90)

    print("Best Model Based on Recall")

    print("=" * 90)

    print(
        f"Model  : {best['Model']}"
    )

    print(
        f"Recall : {best['Recall']:.4f}"
    )


if __name__ == "__main__":

    compare_models()