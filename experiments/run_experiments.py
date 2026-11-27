"""
Batch Experimental Framework
============================
Executes systematic, reproducible comparative experiments across
diverse image classes, collects statistical metrics, and generates
publication-grade figures and CSV datasets.

Produces:
1. experiments/experimental_results.csv
2. experiments/figures/*.png (Rate-distortion, MSE, timing, and rank/quality effects)
"""

import os
import sys
import glob
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.image_loader import load_image
from algorithms.svd_engine import compress_svd_image
from algorithms.dct_engine import compress_dct_image
from evaluation.metrics import compute_all_metrics
from evaluation.storage_analysis import analyze_svd_storage, analyze_dct_storage


def run_full_experimental_suite(
    sample_dir: str = "sample_images",
    output_dir: str = "experiments",
    ranks: list = None,
    qualities: list = None
) -> pd.DataFrame:
    """
    Execute full comparative benchmark across all sample images.
    """
    if ranks is None:
        ranks = [2, 5, 10, 20, 40, 60, 80, 100, 120]
    if qualities is None:
        qualities = [5, 10, 20, 30, 50, 70, 85, 95]

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {sample_dir}")

    all_records = []
    print(f"Starting experiments across {len(image_paths)} images...")

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing Image: {img_name}")
        img_array, (H, W) = load_image(img_path, as_grayscale=True)
        orig_f = img_array.astype(np.float64)
        max_rank = min(H, W)

        # 1. Evaluate SVD across ranks
        print(f"  --> Running SVD benchmark ({len(ranks)} ranks)...")
        for k in ranks:
            if k > max_rank:
                continue
            t0 = time.perf_counter()
            recon, meta = compress_svd_image(img_array, k=k)
            t1 = time.perf_counter()

            metrics = compute_all_metrics(orig_f, recon)
            storage = analyze_svd_storage(img_array.shape, k=k)

            all_records.append({
                "Image_Name": img_name,
                "Image_Dimensions": f"{H}x{W}",
                "Algorithm": "SVD",
                "Parameter": k,
                "Param_Name": "Rank_k",
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "PSNR_dB": metrics["psnr_db"],
                "SSIM": metrics["ssim"],
                "Energy_Retained_Pct": meta["energy_retained_percent"],
                "Sparsity_Pct": 0.0,
                "Entropy_bpc": np.nan,
                "Theoretical_CR": storage["parameter_compression_ratio"],
                "Byte_CR": storage["byte_compression_ratio"],
                "Space_Saving_Pct": storage["parameter_space_saving_percent"],
                "Execution_Time_ms": (t1 - t0) * 1000.0,
                "Decomp_Time_ms": meta["decomposition_time_sec"] * 1000.0,
                "Recon_Time_ms": meta["reconstruction_time_sec"] * 1000.0
            })

        # 2. Evaluate DCT across quality factors
        print(f"  --> Running DCT benchmark ({len(qualities)} qualities)...")
        for q in qualities:
            t0 = time.perf_counter()
            recon, meta = compress_dct_image(img_array, quality=q)
            t1 = time.perf_counter()

            metrics = compute_all_metrics(orig_f, recon)
            storage = analyze_dct_storage(
                img_array.shape,
                meta["quantized_coefficients"],
                meta["shannon_entropy_bpc"]
            )

            all_records.append({
                "Image_Name": img_name,
                "Image_Dimensions": f"{H}x{W}",
                "Algorithm": "DCT",
                "Parameter": q,
                "Param_Name": "Quality_q",
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "PSNR_dB": metrics["psnr_db"],
                "SSIM": metrics["ssim"],
                "Energy_Retained_Pct": np.nan,
                "Sparsity_Pct": storage["sparsity_percent"],
                "Entropy_bpc": storage["shannon_entropy_bpc"],
                "Theoretical_CR": storage["theoretical_compression_ratio"],
                "Byte_CR": storage["coo_sparse_cr"],
                "Space_Saving_Pct": storage["entropy_space_saving_percent"],
                "Execution_Time_ms": (t1 - t0) * 1000.0,
                "Decomp_Time_ms": np.nan,
                "Recon_Time_ms": np.nan
            })

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "experimental_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Experimental dataset saved to: {csv_path} ({len(df)} trials)")

    # Generate Figures
    generate_publication_figures(df, fig_dir)
    return df


def generate_publication_figures(df: pd.DataFrame, output_dir: str):
    """
    Generate all required academic plots and save to PNG.
    """
    print("\nGenerating academic comparison figures...")
    images = df["Image_Name"].unique()

    # 1. Rate-Distortion Curves: PSNR vs Theoretical Compression Ratio
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.5), sharey=True)
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        sub_svd = df[(df["Image_Name"] == img) & (df["Algorithm"] == "SVD")]
        sub_dct = df[(df["Image_Name"] == img) & (df["Algorithm"] == "DCT")]

        ax.plot(sub_svd["Theoretical_CR"], sub_svd["PSNR_dB"], "o-", color="#1f77b4", label="SVD (Param CR)")
        ax.plot(sub_dct["Theoretical_CR"], sub_dct["PSNR_dB"], "s-", color="#d62728", label="DCT (Entropy CR)")
        ax.set_title(img.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Theoretical CR", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("PSNR (dB)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Rate-Distortion Comparison: PSNR vs Compression Ratio", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_rate_distortion_all_images.png"), dpi=200)
    plt.close(fig)

    # 2. MSE Comparison: MSE vs Theoretical CR
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.5), sharey=True)
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        sub_svd = df[(df["Image_Name"] == img) & (df["Algorithm"] == "SVD")]
        sub_dct = df[(df["Image_Name"] == img) & (df["Algorithm"] == "DCT")]

        ax.plot(sub_svd["Theoretical_CR"], sub_svd["MSE"], "o-", color="#1f77b4", label="SVD")
        ax.plot(sub_dct["Theoretical_CR"], sub_dct["MSE"], "s-", color="#d62728", label="DCT")
        ax.set_title(img.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Theoretical CR", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("MSE (Log Scale)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Mean Squared Error Comparison: MSE vs Compression Ratio", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_mse_comparison.png"), dpi=200)
    plt.close(fig)

    # 3. Execution Time Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    timing = df.groupby(["Image_Name", "Algorithm"])["Execution_Time_ms"].mean().unstack()
    timing.plot(kind="bar", ax=ax, color=["#d62728", "#1f77b4"], width=0.6)
    ax.set_title("Mean Execution Time Comparison: SVD vs 8x8 Block DCT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=10)
    ax.set_xlabel("Image Category", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_execution_time_comparison.png"), dpi=200)
    plt.close(fig)

    # 4. Effect of Approximation Rank on SVD Reconstruction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for img in images:
        sub_svd = df[(df["Image_Name"] == img) & (df["Algorithm"] == "SVD")]
        ax1.plot(sub_svd["Parameter"], sub_svd["PSNR_dB"], marker="o", label=img)
        ax2.plot(sub_svd["Parameter"], sub_svd["Energy_Retained_Pct"], marker="s", label=img)

    ax1.set_title("Effect of Rank k on PSNR", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Approximation Rank k", fontsize=10)
    ax1.set_ylabel("PSNR (dB)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.set_title("Effect of Rank k on Energy Retention", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Approximation Rank k", fontsize=10)
    ax2.set_ylabel("Energy Retained (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_svd_rank_effect.png"), dpi=200)
    plt.close(fig)

    # 5. Effect of Quantization Strength on DCT Reconstruction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for img in images:
        sub_dct = df[(df["Image_Name"] == img) & (df["Algorithm"] == "DCT")]
        ax1.plot(sub_dct["Parameter"], sub_dct["PSNR_dB"], marker="o", label=img)
        ax2.plot(sub_dct["Parameter"], sub_dct["Sparsity_Pct"], marker="s", label=img)

    ax1.set_title("Effect of Quality Factor q on PSNR", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Quality Factor q (Higher = Less Quantization)", fontsize=10)
    ax1.set_ylabel("PSNR (dB)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.set_title("Effect of Quality Factor q on Quantized Sparsity", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Quality Factor q", fontsize=10)
    ax2.set_ylabel("Zero Coefficients (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_dct_quality_effect.png"), dpi=200)
    plt.close(fig)

    print(f"[OK] Generated 5 publication figures in: {output_dir}")


if __name__ == "__main__":
    run_full_experimental_suite()
