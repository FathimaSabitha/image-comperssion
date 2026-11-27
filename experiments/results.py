"""
Results Analysis Helper
=======================
Utilities to inspect, aggregate, and format experimental data.
"""

import os
import pandas as pd


def load_experimental_results(csv_path: str = "experiments/experimental_results.csv") -> pd.DataFrame:
    """Load the experimental dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found at: {csv_path}. Run experiments/run_experiments.py first.")
    return pd.read_csv(csv_path)


def get_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a high-level summary comparing SVD and DCT across image categories.
    """
    summary = df.groupby(["Image_Name", "Algorithm"]).agg({
        "PSNR_dB": ["mean", "max"],
        "MSE": ["mean", "min"],
        "Theoretical_CR": ["mean", "max"],
        "Execution_Time_ms": "mean"
    }).round(2)
    return summary
