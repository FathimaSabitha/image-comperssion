"""
Evaluation Metrics Module
=========================
Rigorous statistical and perceptual image quality metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)
- Maximum Absolute Error (L_infinity norm)
"""

from typing import Dict, Any, Union
import numpy as np
from scipy.ndimage import uniform_filter


def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    MSE = (1 / (M*N*C)) * sum (I - K)^2

    Parameters
    ----------
    original : np.ndarray
        Ground truth reference image array.
    reconstructed : np.ndarray
        Reconstructed image array.

    Returns
    -------
    float
        Mean squared error.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(calculate_mse(original, reconstructed)))


def calculate_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    diff = np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))
    return float(np.mean(diff))


def calculate_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_val: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    PSNR = 10 * log10(MAX^2 / MSE)

    Parameters
    ----------
    original : np.ndarray
        Reference image.
    reconstructed : np.ndarray
        Reconstructed image.
    max_val : float, default=255.0
        Maximum possible pixel value (255.0 for 8-bit unsigned images).

    Returns
    -------
    float
        PSNR in dB. If MSE == 0, returns 100.0 dB (practical infinity).
    """
    mse = calculate_mse(original, reconstructed)
    if mse <= 1e-12:
        return 100.0  # Perfect reconstruction representation
    
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def calculate_max_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate maximum absolute pixel difference (L_infinity norm)."""
    diff = np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))
    return float(np.max(diff))


def calculate_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_val: float = 255.0,
    win_size: int = 7
) -> float:
    """
    Calculate Structural Similarity Index Measure (SSIM).
    Uses local mean, variance, and covariance over a sliding window.

    Parameters
    ----------
    original : np.ndarray
        Reference image (2D grayscale or 3D color).
    reconstructed : np.ndarray
        Reconstructed image.
    max_val : float, default=255.0
        Maximum dynamic range.
    win_size : int, default=7
        Size of sliding window (must be odd).

    Returns
    -------
    float
        SSIM value in [-1.0, 1.0], where 1.0 indicates perfect structural identity.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

    # If color image, compute SSIM per channel and average
    if original.ndim == 3:
        num_channels = original.shape[2]
        ssim_vals = [
            calculate_ssim(original[:, :, c], reconstructed[:, :, c], max_val, win_size)
            for c in range(num_channels)
        ]
        return float(np.mean(ssim_vals))

    X = original.astype(np.float64)
    Y = reconstructed.astype(np.float64)

    # SSIM stability constants
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    # Compute local means
    mu1 = uniform_filter(X, size=win_size)
    mu2 = uniform_filter(Y, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = uniform_filter(X ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(Y ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(X * Y, size=win_size) - mu1_mu2

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_val: float = 255.0
) -> Dict[str, float]:
    """
    Compute a full suite of statistical and perceptual evaluation metrics.
    """
    mse = calculate_mse(original, reconstructed)
    rmse = calculate_rmse(original, reconstructed)
    mae = calculate_mae(original, reconstructed)
    psnr = calculate_psnr(original, reconstructed, max_val=max_val)
    ssim = calculate_ssim(original, reconstructed, max_val=max_val)
    max_err = calculate_max_error(original, reconstructed)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr_db": psnr,
        "ssim": ssim,
        "max_error": max_err
    }
