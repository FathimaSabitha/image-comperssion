"""
DCT Compression Engine
======================
Mathematical Implementation of 8x8 Block-Based 2D Discrete Cosine Transform (DCT-II),
Psychovisual Quantization, Inverse DCT (IDCT), and Entropy Modeling.

Theory:
-------
For an 8x8 spatial block X:
    Forward DCT:  F = T * X * T^T
    Quantization: F_Q = round(F / Q_scaled)
    Dequant:      F_hat = F_Q * Q_scaled
    Inverse DCT:  X_hat = T^T * F_hat * T

Where:
    - T in R^{8x8} is the orthonormal DCT-II transformation matrix.
    - Level shifting (-128.0) is applied prior to forward DCT to center intensities.
    - Quantization attenuates high spatial frequencies according to Human Visual System (HVS).
"""

from typing import Tuple, Dict, Any, List, Optional
import time
import numpy as np
from processing.preprocessing import block_partition_2d, block_stitch_2d
from processing.image_loader import pad_to_block_size, crop_to_original_size


# Standard JPEG Luminance Quantization Matrix (ITU-T T.81 / ISO/IEC 10918-1)
JPEG_LUMINANCE_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

# Standard JPEG Chrominance Quantization Matrix
JPEG_CHROMINANCE_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float64)

# Zigzag scan order indices for 8x8 block (frequency ascending)
ZIGZAG_ORDER = [
    (0, 0),
    (0, 1), (1, 0),
    (2, 0), (1, 1), (0, 2),
    (0, 3), (1, 2), (2, 1), (3, 0),
    (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
    (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
    (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
    (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
    (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
    (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
    (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
    (4, 7), (5, 6), (6, 5), (7, 4),
    (7, 5), (6, 6), (5, 7),
    (6, 7), (7, 6),
    (7, 7)
]


def generate_dct_basis_matrix(block_size: int = 8) -> np.ndarray:
    """
    Construct the orthonormal 1D DCT-II transform matrix T of shape (N, N).
    
    Formula:
        T[u, x] = 1 / sqrt(N)                       for u = 0
        T[u, x] = sqrt(2 / N) * cos((2x + 1)u*pi / (2N)) for u > 0
    """
    N = block_size
    T = np.zeros((N, N), dtype=np.float64)
    c0 = 1.0 / np.sqrt(N)
    c1 = np.sqrt(2.0 / N)

    for u in range(N):
        alpha = c0 if u == 0 else c1
        for x in range(N):
            T[u, x] = alpha * np.cos((2.0 * x + 1.0) * u * np.pi / (2.0 * N))

    return T


# Precompute standard 8x8 DCT transform matrix
DCT_MATRIX_8X8 = generate_dct_basis_matrix(8)


def scale_quantization_matrix(base_table: np.ndarray, quality: int) -> np.ndarray:
    """
    Scale quantization table using the standard Independent JPEG Group (IJG) formula.

    Parameters
    ----------
    base_table : np.ndarray
        8x8 baseline quantization table.
    quality : int
        Quality factor in [1, 100]. Higher = better quality / less quantization.

    Returns
    -------
    scaled_table : np.ndarray
        8x8 integer quantization table with values clipped to [1, 255].
    """
    quality = max(1, min(100, int(quality)))

    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    scaled = np.floor((base_table * scale + 50.0) / 100.0)
    # Clip to avoid zero division and uint8 overflow
    scaled = np.clip(scaled, 1.0, 255.0)
    return scaled.astype(np.float64)


def forward_dct_block(block: np.ndarray, T: np.ndarray = DCT_MATRIX_8X8) -> np.ndarray:
    """
    Compute 2D forward DCT of an 8x8 block: F = T * (block - 128) * T^T
    """
    level_shifted = block.astype(np.float64) - 128.0
    # Matrix multiplication: T @ level_shifted @ T^T
    return T @ level_shifted @ T.T


def inverse_dct_block(F: np.ndarray, T: np.ndarray = DCT_MATRIX_8X8) -> np.ndarray:
    """
    Compute 2D inverse DCT of an 8x8 frequency block: block = (T^T * F * T) + 128
    """
    spatial = T.T @ F @ T
    return spatial + 128.0


def quantize_block(F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Quantize frequency coefficients: F_Q = round(F / Q)
    """
    return np.round(F / Q).astype(np.float64)


def dequantize_block(F_Q: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Dequantize frequency coefficients: F_hat = F_Q * Q
    """
    return F_Q * Q


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    Convert an 8x8 matrix into a 1D vector of 64 elements using zigzag traversal.
    """
    return np.array([block[r, c] for (r, c) in ZIGZAG_ORDER], dtype=block.dtype)


def calculate_entropy(quantized_coeffs: np.ndarray) -> float:
    """
    Calculate first-order Shannon Entropy of quantized integer coefficients.
    H = - sum p_i * log2(p_i) in bits per coefficient.
    """
    flat = quantized_coeffs.flatten().astype(np.int64)
    if flat.size == 0:
        return 0.0

    _, counts = np.unique(flat, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def compress_dct_channel_2d(
    channel_2d: np.ndarray,
    quality: int = 50,
    is_chroma: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compress a single 2D channel using 8x8 block DCT and quantization.

    Parameters
    ----------
    channel_2d : np.ndarray
        2D image channel of shape (M, N). Assumed to be padded to multiples of 8.
    quality : int
        Quality factor in [1, 100].
    is_chroma : bool
        If True, uses chrominance quantization table; else luminance table.

    Returns
    -------
    reconstructed_2d : np.ndarray
        Reconstructed spatial channel of shape (M, N).
    quantized_blocks : np.ndarray
        4D array of quantized coefficients of shape (num_h, num_w, 8, 8).
    channel_meta : dict
        Sparsity, entropy, and timing for this channel.
    """
    t0 = time.perf_counter()
    base_table = JPEG_CHROMINANCE_QUANT_TABLE if is_chroma else JPEG_LUMINANCE_QUANT_TABLE
    Q = scale_quantization_matrix(base_table, quality)
    T = DCT_MATRIX_8X8

    # 1. Partition into 8x8 spatial blocks
    blocks = block_partition_2d(channel_2d, block_size=8)
    num_h, num_w = blocks.shape[:2]

    quantized_blocks = np.zeros_like(blocks, dtype=np.float64)
    reconstructed_blocks = np.zeros_like(blocks, dtype=np.float64)

    # Vectorized block DCT across all blocks:
    # blocks has shape (num_h, num_w, 8, 8)
    # Forward DCT: F = T @ (blocks - 128) @ T.T
    level_shifted = blocks.astype(np.float64) - 128.0
    F = np.matmul(np.matmul(T, level_shifted), T.T)

    # Quantization
    F_Q = np.round(F / Q)
    quantized_blocks = F_Q

    # Dequantization
    F_hat = F_Q * Q

    # Inverse DCT: X_hat = T.T @ F_hat @ T + 128
    X_hat = np.matmul(np.matmul(T.T, F_hat), T) + 128.0
    reconstructed_blocks = X_hat

    # 2. Reassemble blocks into continuous 2D matrix
    reconstructed_2d = block_stitch_2d(reconstructed_blocks)
    t1 = time.perf_counter()

    # Metrics
    total_coeffs = quantized_blocks.size
    zero_coeffs = np.count_nonzero(quantized_blocks == 0)
    sparsity_percent = (zero_coeffs / total_coeffs) * 100.0 if total_coeffs > 0 else 0.0
    entropy = calculate_entropy(quantized_blocks)

    channel_meta = {
        "execution_time_sec": t1 - t0,
        "total_coefficients": total_coeffs,
        "zero_coefficients": zero_coeffs,
        "sparsity_percent": sparsity_percent,
        "shannon_entropy_bpc": entropy,
        "quantization_matrix": Q
    }

    return reconstructed_2d, quantized_blocks, channel_meta


def compress_dct_image(
    image_matrix: np.ndarray,
    quality: int = 50
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compress 2D grayscale or 3D color image using 8x8 block-based DCT.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image array in uint8 or float, shape (M, N) or (M, N, C).
    quality : int
        Quality factor in [1, 100].

    Returns
    -------
    reconstructed : np.ndarray
        Reconstructed image array in float64 [0.0, 255.0] matching original dimensions.
    metadata : dict
        Quality factor, sparsity, entropy, theoretical CR, and execution time.
    """
    t_start = time.perf_counter()
    orig_dims = (image_matrix.shape[0], image_matrix.shape[1])
    is_color = (image_matrix.ndim == 3)

    # Pad image to multiples of 8
    padded_img, (pad_h, pad_w) = pad_to_block_size(image_matrix, block_size=8, mode="edge")

    if not is_color:
        recon_padded, q_blocks, meta = compress_dct_channel_2d(
            padded_img.astype(np.float64), quality=quality, is_chroma=False
        )
        reconstructed = crop_to_original_size(recon_padded, orig_dims)
        t_end = time.perf_counter()

        # Theoretical Compression Ratio based on First-Order Shannon Entropy
        # 8 bits / pixel baseline vs entropy (bits/pixel)
        entropy = meta["shannon_entropy_bpc"]
        # Safe compression ratio (entropy lower bounded by 0.1 to avoid div by zero)
        eff_entropy = max(entropy, 0.05)
        cr_entropy = 8.0 / eff_entropy

        meta.update({
            "quality": quality,
            "total_time_sec": t_end - t_start,
            "original_dims": orig_dims,
            "padded_dims": padded_img.shape[:2],
            "padding_added": (pad_h, pad_w),
            "theoretical_cr_entropy": cr_entropy,
            "is_color": False,
            "quantized_coefficients": q_blocks
        })
        return reconstructed, meta

    else:
        # Multi-channel RGB
        num_channels = image_matrix.shape[2]
        recon_channels = []
        all_q_blocks = []
        total_zeros = 0
        total_coeffs = 0
        channel_metas = []

        for c in range(num_channels):
            # Channel 0 is R (or Y), 1 is G (or Cb), 2 is B (or Cr)
            is_chroma = (c > 0)
            recon_ch, q_b, ch_m = compress_dct_channel_2d(
                padded_img[:, :, c].astype(np.float64), quality=quality, is_chroma=is_chroma
            )
            recon_channels.append(recon_ch)
            all_q_blocks.append(q_b)
            total_zeros += ch_m["zero_coefficients"]
            total_coeffs += ch_m["total_coefficients"]
            channel_metas.append(ch_m)

        recon_padded = np.stack(recon_channels, axis=-1)
        reconstructed = crop_to_original_size(recon_padded, orig_dims)
        t_end = time.perf_counter()

        all_q_blocks_arr = np.stack(all_q_blocks, axis=0)
        overall_entropy = calculate_entropy(all_q_blocks_arr)
        overall_sparsity = (total_zeros / total_coeffs) * 100.0 if total_coeffs > 0 else 0.0
        eff_entropy = max(overall_entropy, 0.05)
        cr_entropy = 8.0 / eff_entropy

        metadata = {
            "quality": quality,
            "total_time_sec": t_end - t_start,
            "original_dims": orig_dims,
            "padded_dims": padded_img.shape[:2],
            "padding_added": (pad_h, pad_w),
            "total_coefficients": total_coeffs,
            "zero_coefficients": total_zeros,
            "sparsity_percent": overall_sparsity,
            "shannon_entropy_bpc": overall_entropy,
            "theoretical_cr_entropy": cr_entropy,
            "is_color": True,
            "quantized_coefficients": all_q_blocks_arr,
            "channel_details": channel_metas
        }
        return reconstructed, metadata


def verify_dct_on_small_block() -> Dict[str, Any]:
    """
    Mathematical verification on known 8x8 test blocks.
    Validates:
    1. Orthonormality of DCT matrix: T * T^T = I
    2. Uniform block test: constant 128 block yields zero AC coefficients.
    3. Perfect invertibility: unquantized IDCT(DCT(X)) == X within float precision.
    """
    T = DCT_MATRIX_8X8
    # 1. Orthonormality check
    ortho_error = float(np.linalg.norm(T @ T.T - np.eye(8), "fro"))

    # 2. Uniform block
    constant_block = np.full((8, 8), 128.0)
    F_const = forward_dct_block(constant_block, T)
    const_ac_energy = float(np.sum(F_const ** 2))  # Level shifted 128 - 128 = 0

    # 3. Arbitrary test block with spatial ramp
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    test_block = (x * 15.0 + y * 12.0).astype(np.float64)

    F = forward_dct_block(test_block, T)
    reconstructed_exact = inverse_dct_block(F, T)
    exact_recon_error = float(np.linalg.norm(test_block - reconstructed_exact, "fro"))

    # 4. Quantization test with quality 50
    Q50 = scale_quantization_matrix(JPEG_LUMINANCE_QUANT_TABLE, 50)
    F_Q = quantize_block(F, Q50)
    F_hat = dequantize_block(F_Q, Q50)
    reconstructed_lossy = inverse_dct_block(F_hat, T)
    lossy_recon_error = float(np.linalg.norm(test_block - reconstructed_lossy, "fro"))

    return {
        "orthonormality_frobenius_error": ortho_error,
        "is_orthonormal": ortho_error < 1e-12,
        "constant_block_ac_energy": const_ac_energy,
        "exact_inversion_error": exact_recon_error,
        "is_invertible": exact_recon_error < 1e-12,
        "lossy_reconstruction_error": lossy_recon_error,
        "quantized_zeros_count": int(np.count_nonzero(F_Q == 0))
    }
