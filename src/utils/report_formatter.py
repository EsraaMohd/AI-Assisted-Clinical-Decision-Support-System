def format_summary(summary: dict) -> str:
    """
    Format the dataset summary as a Markdown report.

    Args:
        summary: Dictionary containing dataset summary.

    Returns:
        Formatted Markdown string.
    """

    report = []

    report.append("# Dataset Summary\n")

    report.append("## Dataset Shape")
    report.append(f"- **Rows:** {summary['shape'][0]}")
    report.append(f"- **Columns:** {summary['shape'][1]}\n")

    report.append("## Columns")

    for column in summary["columns"]:
        report.append(f"- {column}")

    report.append("\n## Data Types\n")

    report.append("| Column | Data Type |")
    report.append("|--------|-----------|")

    for column, dtype in summary["data_types"].items():
        report.append(f"| {column} | {dtype} |")

    report.append("\n## Missing Values\n")

    report.append("| Column | Missing Values |")
    report.append("|--------|----------------|")

    for column, missing in summary["missing_values"].items():
        report.append(f"| {column} | {missing} |")

    report.append("\n## Duplicate Rows")

    report.append(f"- {summary['duplicate_rows']}")

    return "\n".join(report)