"""
Storage and Complexity Analysis Module
======================================
Provides formal mathematical storage models distinguishing theoretical
representation parameters, information-theoretic entropy, and byte-level storage.
"""

from typing import Dict, Any, Tuple
import numpy as np


def analyze_svd_storage(
    image_shape: Tuple[int, ...],
    k: int,
    float_bytes: int = 4
) -> Dict[str, Any]:
    """
    Perform theoretical and byte-level storage analysis for SVD rank-k representation.

    Parameters
    ----------
    image_shape : tuple
        (M, N) for grayscale or (M, N, C) for color.
    k : int
        Approximation rank.
    float_bytes : int, default=4
        Bytes per float parameter (4 for single precision float32, 8 for float64).

    Returns
    -------
    dict
        Comprehensive storage metrics.
    """
    M, N = image_shape[:2]
    num_channels = 1 if len(image_shape) == 2 else image_shape[2]
    
    # 1. Parameter Model
    raw_parameters = M * N * num_channels
    svd_parameters = num_channels * k * (M + N + 1)
    param_cr = float(raw_parameters / svd_parameters) if svd_parameters > 0 else 0.0

    # 2. Critical Breakeven Rank (Parameter Model)
    breakeven_rank_param = int((M * N) / (M + N + 1))

    # 3. Byte-Level Model (Raw 8-bit image = 1 byte/pixel vs SVD float32/float64)
    raw_bytes = raw_parameters * 1  # 8-bit unsigned int
    svd_bytes = svd_parameters * float_bytes
    byte_cr = float(raw_bytes / svd_bytes) if svd_bytes > 0 else 0.0
    breakeven_rank_bytes = int((M * N) / (float_bytes * (M + N + 1)))

    return {
        "raw_parameter_count": raw_parameters,
        "svd_parameter_count": svd_parameters,
        "parameter_compression_ratio": param_cr,
        "parameter_space_saving_percent": max(0.0, (1.0 - (svd_parameters / raw_parameters)) * 100.0),
        "breakeven_rank_parameters": breakeven_rank_param,
        "raw_bytes": raw_bytes,
        "svd_bytes": svd_bytes,
        "byte_compression_ratio": byte_cr,
        "breakeven_rank_bytes": breakeven_rank_bytes,
        "is_param_compressed": param_cr > 1.0,
        "is_byte_compressed": byte_cr > 1.0,
        "k_selected": k,
        "max_possible_rank": min(M, N)
    }


def analyze_dct_storage(
    image_shape: Tuple[int, ...],
    quantized_blocks: np.ndarray,
    entropy_bpc: float
) -> Dict[str, Any]:
    """
    Perform theoretical storage analysis for quantized DCT representation.

    Parameters
    ----------
    image_shape : tuple
        Original image shape (M, N) or (M, N, C).
    quantized_blocks : np.ndarray
        Array of quantized integer coefficients.
    entropy_bpc : float
        First-order Shannon entropy in bits per coefficient.

    Returns
    -------
    dict
        Entropy and sparsity storage metrics.
    """
    total_coeffs = quantized_blocks.size
    zero_coeffs = int(np.count_nonzero(quantized_blocks == 0))
    non_zero_coeffs = total_coeffs - zero_coeffs
    sparsity_percent = (zero_coeffs / total_coeffs) * 100.0 if total_coeffs > 0 else 0.0

    # Raw 8-bit pixel uncompressed bits
    raw_bits = total_coeffs * 8
    
    # Information-theoretic lower bound on storage (Shannon Source Coding Theorem)
    safe_entropy = max(entropy_bpc, 0.05)
    theoretical_entropy_bits = total_coeffs * safe_entropy
    theoretical_cr = 8.0 / safe_entropy

    # Sparse Coordinate List (COO) representation model:
    # Storing non-zero coefficients: value (int16: 2 bytes) + 2D index (uint16: 2 bytes)
    coo_bytes = non_zero_coeffs * 4
    raw_bytes = total_coeffs * 1
    coo_cr = float(raw_bytes / coo_bytes) if coo_bytes > 0 else 0.0

    return {
        "total_coefficients": total_coeffs,
        "zero_coefficients": zero_coeffs,
        "non_zero_coefficients": non_zero_coeffs,
        "sparsity_percent": sparsity_percent,
        "shannon_entropy_bpc": entropy_bpc,
        "raw_bits": raw_bits,
        "theoretical_compressed_bits": theoretical_entropy_bits,
        "theoretical_compression_ratio": theoretical_cr,
        "entropy_space_saving_percent": max(0.0, (1.0 - (1.0 / theoretical_cr)) * 100.0),
        "coo_sparse_cr": coo_cr
    }
