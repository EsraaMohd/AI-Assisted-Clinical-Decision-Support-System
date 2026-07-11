from pathlib import Path


def save_report(report: str, output_path: Path) -> None:
    """
    Save a report to a text or Markdown file.

    Args:
        report: Report content.
        output_path: Output file path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)