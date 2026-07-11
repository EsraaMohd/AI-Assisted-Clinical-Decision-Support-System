from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(
        
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save the target variable distribution.

    Args:
        df: Input dataset.
        output_dir: Directory to save the figure.
    """

    target_counts = df["readmitted"].value_counts()

    plt.figure(figsize=(6, 4))

    target_counts.plot(kind="bar")

    plt.title("Target Variable Distribution")
    plt.xlabel("Readmission")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "target_distribution.png")

    plt.close()


def plot_age_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save the age distribution.
    """

    plt.figure(figsize=(8, 5))

    df["age"].value_counts().sort_index().plot(kind="bar")

    plt.title("Age Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "age_distribution.png")

    plt.close()

def plot_time_in_hospital_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save the hospital stay distribution.
    """

    plt.figure(figsize=(8, 5))

    df["time_in_hospital"].hist(bins=14)

    plt.title("Hospital Stay Distribution")
    plt.xlabel("Days")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "hospital_stay_distribution.png")

    plt.close()

def plot_time_in_hospital_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save the hospital stay distribution.
    """

    plt.figure(figsize=(8, 5))

    df["time_in_hospital"].hist(bins=14)

    plt.title("Hospital Stay Distribution")
    plt.xlabel("Days")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "hospital_stay_distribution.png")

    plt.close()

def plot_medications_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save the medications distribution.
    """

    plt.figure(figsize=(8, 5))

    df["n_medications"].hist(bins=20)

    plt.title("Medication Distribution")
    plt.xlabel("Number of Medications")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "medications_distribution.png")

    plt.close()                