"""
Visualization Module
====================
Publication-quality Matplotlib figures for mathematical and experimental analysis:
- Singular value spectrum (scree plot & energy accumulation)
- 2D DCT-II basis function dictionary visualization (8x8 grid)
- Residual error maps (|Original - Reconstructed|)
- Rate-Distortion curves (PSNR vs CR, MSE vs Parameter)
- Execution time and computational complexity plots
"""

from typing import Optional, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit and servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from algorithms.dct_engine import DCT_MATRIX_8X8


def plot_singular_value_decay(
    s: np.ndarray,
    k_selected: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Generate a two-panel plot:
    Panel 1: Singular value decay (log-scale scree plot).
    Panel 2: Cumulative energy retention curve with threshold markers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ranks = np.arange(1, len(s) + 1)
    # Energy
    cum_energy = (np.cumsum(s ** 2) / np.sum(s ** 2)) * 100.0

    # Panel 1: Scree Plot (Log Scale)
    ax1.plot(ranks, s, color="#1f77b4", linewidth=2.0, label=r"Singular Values $\sigma_i$")
    ax1.set_yscale("log")
    ax1.set_title("Singular Value Spectrum (Scree Plot)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Singular Value Index $i$", fontsize=10)
    ax1.set_ylabel(r"Magnitude $\sigma_i$ (Log Scale)", fontsize=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    if k_selected is not None and 1 <= k_selected <= len(s):
        ax1.axvline(x=k_selected, color="#d62728", linestyle="--", label=f"Selected Rank $k={k_selected}$")
        ax1.scatter([k_selected], [s[k_selected - 1]], color="#d62728", s=40, zorder=5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Cumulative Energy Retention
    ax2.plot(ranks, cum_energy, color="#2ca02c", linewidth=2.0, label="Cumulative Energy $\mathcal{E}(k)$")
    ax2.axhline(y=90.0, color="#ff7f0e", linestyle=":", label="90% Energy")
    ax2.axhline(y=99.0, color="#d62728", linestyle=":", label="99% Energy")
    ax2.set_title("Cumulative Energy Retention", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Rank $k$", fontsize=10)
    ax2.set_ylabel("Retained Energy (%)", fontsize=10)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle="--", alpha=0.5)

    if k_selected is not None and 1 <= k_selected <= len(s):
        retained = cum_energy[k_selected - 1]
        ax2.axvline(x=k_selected, color="#d62728", linestyle="--")
        ax2.scatter([k_selected], [retained], color="#d62728", s=40, zorder=5)
        ax2.annotate(f"{retained:.1f}%", xy=(k_selected, retained), xytext=(k_selected + 3, retained - 8),
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_dct_basis_functions(block_size: int = 8, figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Visualizes the 64 elementary 2D orthonormal basis functions of the 8x8 DCT-II.
    Basis block B(u, v) = T[u, :]^T * T[v, :]
    """
    T = DCT_MATRIX_8X8
    fig, axes = plt.subplots(block_size, block_size, figsize=figsize)
    fig.suptitle(r"8$\times$8 Discrete Cosine Transform (DCT-II) Orthonormal Basis Dictionary",
                 fontsize=12, fontweight="bold", y=0.98)

    for u in range(block_size):
        for v in range(block_size):
            # Outer product of row u and row v of transform matrix T
            basis_uv = np.outer(T[u, :], T[v, :])
            ax = axes[u, v]
            ax.imshow(basis_uv, cmap="gray", vmin=-0.35, vmax=0.35)
            ax.set_xticks([])
            ax.set_yticks([])
            if u == 0:
                ax.set_title(f"v={v}", fontsize=7)
            if v == 0:
                ax.set_ylabel(f"u={u}", fontsize=7)

    plt.tight_layout()
    return fig


def plot_residual_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction Residual Error Map (|Original - Reconstructed|)",
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    Compute and plot absolute pixel error heatmap |I(x, y) - K(x, y)|.
    """
    orig_f = original.astype(np.float64)
    recon_f = reconstructed.astype(np.float64)

    if original.ndim == 3:
        # Mean across color channels
        error_map = np.mean(np.abs(orig_f - recon_f), axis=-1)
    else:
        error_map = np.abs(orig_f - recon_f)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(error_map, cmap="inferno", vmin=0, vmax=max(30.0, float(np.max(error_map))))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Pixel Error [0-255]", fontsize=9)
    plt.tight_layout()
    return fig


def plot_rate_distortion_curves(
    svd_df: pd.DataFrame,
    dct_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot comparative Rate-Distortion curves:
    Panel 1: PSNR (dB) vs Theoretical Compression Ratio.
    Panel 2: MSE vs Compression Ratio.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: PSNR vs Compression Ratio
    ax1.plot(
        svd_df["Param_CR"], svd_df["PSNR_dB"],
        marker="o", color="#1f77b4", linewidth=2.0, label="SVD (Parameter CR)"
    )
    ax1.plot(
        dct_df["Theoretical_CR"], dct_df["PSNR_dB"],
        marker="s", color="#d62728", linewidth=2.0, label="DCT (Entropy CR)"
    )
    ax1.set_title("Rate-Distortion Curve: PSNR vs Compression Ratio", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Theoretical Compression Ratio (Higher is more compressed)", fontsize=10)
    ax1.set_ylabel("PSNR (dB) [Higher is better]", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", framealpha=0.9)

    # Panel 2: MSE vs Compression Ratio
    ax2.plot(
        svd_df["Param_CR"], svd_df["MSE"],
        marker="o", color="#1f77b4", linewidth=2.0, label="SVD (Parameter CR)"
    )
    ax2.plot(
        dct_df["Theoretical_CR"], dct_df["MSE"],
        marker="s", color="#d62728", linewidth=2.0, label="DCT (Entropy CR)"
    )
    ax2.set_title("Error vs Compression Ratio: MSE vs CR", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Theoretical Compression Ratio", fontsize=10)
    ax2.set_ylabel("Mean Squared Error (MSE) [Lower is better]", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_execution_time_comparison(
    svd_df: pd.DataFrame,
    dct_df: pd.DataFrame,
    figsize: Tuple[int, int] = (7, 4)
) -> plt.Figure:
    """
    Bar chart comparing average execution times of SVD vs DCT.
    """
    fig, ax = plt.subplots(figsize=figsize)
    svd_mean_ms = svd_df["Total_Time_ms"].mean()
    dct_mean_ms = dct_df["Total_Time_ms"].mean()

    bars = ax.bar(
        ["SVD (Rank Sweep Avg)", "DCT (Block 8x8 Avg)"],
        [svd_mean_ms, dct_mean_ms],
        color=["#1f77b4", "#d62728"],
        width=0.45
    )
    ax.set_title("Computational Cost (Execution Time)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.2f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_quantized_energy_distribution(
    quantized_blocks: np.ndarray,
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    Plot 8x8 heatmap showing the mean absolute magnitude of coefficients across all blocks.
    Demonstrates energy compaction into the top-left DC/low-frequency corner.
    """
    # quantized_blocks shape: (num_h, num_w, 8, 8) or (C, num_h, num_w, 8, 8)
    if quantized_blocks.ndim == 5:
        # Mean over channels
        mean_coeffs = np.mean(np.abs(quantized_blocks), axis=(0, 1, 2))
    elif quantized_blocks.ndim == 4:
        mean_coeffs = np.mean(np.abs(quantized_blocks), axis=(0, 1))
    else:
        raise ValueError(f"Expected 4D or 5D array, got {quantized_blocks.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mean_coeffs, cmap="viridis")
    ax.set_title("Mean Absolute Quantized DCT Coefficients (8x8)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Horizontal Frequency Index $v$", fontsize=10)
    ax.set_ylabel("Vertical Frequency Index $u$", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Magnitude", fontsize=9)

    for i in range(8):
        for j in range(8):
            val = mean_coeffs[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color="white" if val < np.max(mean_coeffs) * 0.5 else "black", fontsize=7)

    plt.tight_layout()
    return fig
