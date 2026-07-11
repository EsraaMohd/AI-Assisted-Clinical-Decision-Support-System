from pathlib import Path

from src.analysis.eda import plot_age_distribution, plot_medications_distribution, plot_target_distribution, plot_time_in_hospital_distribution
from src.data.load_data import load_dataset


def main():

    dataset_path = Path("data/raw/hospital_readmissions.csv")

    output_dir = Path("images/eda")

    df = load_dataset(dataset_path)

    plot_target_distribution(df, output_dir)
    plot_age_distribution(df, output_dir)
    plot_time_in_hospital_distribution(df, output_dir)
    plot_medications_distribution(df, output_dir)

if __name__ == "__main__":
    main()