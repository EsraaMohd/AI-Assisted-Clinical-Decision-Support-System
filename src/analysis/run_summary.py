from pathlib import Path

from src.analysis.dataset_summary import generate_summary
from src.data.load_data import load_dataset
from src.utils.file_manager import save_report
from src.utils.report_formatter import format_summary


def main():

    dataset_path = Path("data/raw/hospital_readmissions.csv")

    report_path = Path("reports/summary.md")

    df = load_dataset(dataset_path)

    summary = generate_summary(df)

    markdown = format_summary(summary)

    save_report(markdown, report_path)

    print("Dataset summary report generated successfully.")


if __name__ == "__main__":
    main()