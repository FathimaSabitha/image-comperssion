"""
Benchmarking Pipeline Module
============================
Automated sweep evaluator for varying SVD ranks and DCT quality factors.
Outputs structured Pandas DataFrames for rate-distortion analysis.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from algorithms.svd_engine import compress_svd_image
from algorithms.dct_engine import compress_dct_image
from evaluation.metrics import compute_all_metrics
from evaluation.storage_analysis import analyze_svd_storage, analyze_dct_storage


def generate_default_svd_ranks(max_rank: int, num_points: int = 12) -> List[int]:
    """Generate geometrically and linearly spaced test ranks up to max_rank."""
    ranks = {1, 2, 4, 8, 16, 32, 64, 128, max_rank // 4, max_rank // 2, max_rank}
    # Keep only valid within [1, max_rank]
    filtered = sorted([int(r) for r in ranks if 1 <= r <= max_rank])
    return filtered


def generate_default_dct_qualities() -> List[int]:
    """Standard quality points across the JPEG quality curve."""
    return [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def benchmark_svd_ranks(
    image_matrix: np.ndarray,
    ranks: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Run parametric sweep across SVD ranks k.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image.
    ranks : list of int, optional
        Ranks to evaluate. Defaults to sensible logarithmic/linear spread.

    Returns
    -------
    pd.DataFrame
        Table with rank, metrics, energy %, CR, and runtime.
    """
    M, N = image_matrix.shape[:2]
    max_k = min(M, N)
    if ranks is None:
        ranks = generate_default_svd_ranks(max_k)

    records = []
    # Pre-convert to float for ground truth
    orig_f = image_matrix.astype(np.float64)

    for k in ranks:
        k_clamped = max(1, min(k, max_k))
        recon, meta = compress_svd_image(image_matrix, k=k_clamped)
        metrics = compute_all_metrics(orig_f, recon)
        storage = analyze_svd_storage(image_matrix.shape, k=k_clamped)

        records.append({
            "Algorithm": "SVD",
            "Parameter_k": k_clamped,
            "MSE": metrics["mse"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "PSNR_dB": metrics["psnr_db"],
            "SSIM": metrics["ssim"],
            "Energy_Percent": meta["energy_retained_percent"],
            "Param_CR": storage["parameter_compression_ratio"],
            "Byte_CR": storage["byte_compression_ratio"],
            "Space_Saving_Pct": storage["parameter_space_saving_percent"],
            "Total_Time_ms": meta["total_time_sec"] * 1000.0,
            "Decomp_Time_ms": meta["decomposition_time_sec"] * 1000.0,
            "Recon_Time_ms": meta["reconstruction_time_sec"] * 1000.0
        })

    return pd.DataFrame(records)


def benchmark_dct_qualities(
    image_matrix: np.ndarray,
    qualities: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Run parametric sweep across DCT quality factors q.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image.
    qualities : list of int, optional
        Quality factors in [1, 100].

    Returns
    -------
    pd.DataFrame
        Table with quality, metrics, sparsity, entropy, CR, and runtime.
    """
    if qualities is None:
        qualities = generate_default_dct_qualities()

    records = []
    orig_f = image_matrix.astype(np.float64)

    for q in qualities:
        recon, meta = compress_dct_image(image_matrix, quality=q)
        metrics = compute_all_metrics(orig_f, recon)
        storage = analyze_dct_storage(
            image_matrix.shape,
            meta["quantized_coefficients"],
            meta["shannon_entropy_bpc"]
        )

        records.append({
            "Algorithm": "DCT",
            "Quality_q": q,
            "MSE": metrics["mse"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "PSNR_dB": metrics["psnr_db"],
            "SSIM": metrics["ssim"],
            "Sparsity_Percent": storage["sparsity_percent"],
            "Entropy_bpc": storage["shannon_entropy_bpc"],
            "Theoretical_CR": storage["theoretical_compression_ratio"],
            "COO_Sparse_CR": storage["coo_sparse_cr"],
            "Space_Saving_Pct": storage["entropy_space_saving_percent"],
            "Total_Time_ms": meta["total_time_sec"] * 1000.0
        })

    return pd.DataFrame(records)
