"""
SVD Compression Engine
======================
Mathematical Implementation of Singular Value Decomposition (SVD)
and Eckart-Young-Mirsky Low-Rank Matrix Approximation.

Theory:
-------
For any real matrix A in R^{M x N}:
    A = U * Sigma * V^T = sum_{i=1}^r sigma_i * (u_i * v_i^T)

Where:
    - U in R^{M x M} contains orthonormal eigenvectors of A * A^T
    - V in R^{N x N} contains orthonormal eigenvectors of A^T * A
    - Sigma in R^{M x N} has non-negative singular values sigma_1 >= sigma_2 >= ... >= 0
    - Truncated rank-k approximation A_k = U_k * Sigma_k * V_k^T
    - Eckart-Young-Mirsky Theorem guarantees ||A - A_k||_F is minimized globally.
"""

from typing import Tuple, Dict, Any, Union, Optional
import time
import numpy as np
import scipy.linalg


def compute_svd_thin(matrix_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thin (economy-size) Singular Value Decomposition.

    Parameters
    ----------
    matrix_2d : np.ndarray
        Real 2D matrix A of shape (M, N).

    Returns
    -------
    U : np.ndarray
        Left singular vectors of shape (M, min(M, N)).
    s : np.ndarray
        Singular values in descending order of length min(M, N).
    Vt : np.ndarray
        Right singular vectors (transposed) of shape (min(M, N), N).
    """
    if matrix_2d.ndim != 2:
        raise ValueError(f"compute_svd_thin requires 2D matrix, got shape {matrix_2d.shape}")
    
    # scipy.linalg.svd with full_matrices=False computes economy SVD
    U, s, Vt = scipy.linalg.svd(matrix_2d, full_matrices=False)
    return U, s, Vt


def reconstruct_svd_rank_k(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Reconstruct rank-k approximation A_k = U_k * Sigma_k * V_k^T.

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors of shape (M, K_max).
    s : np.ndarray
        Singular values of shape (K_max,).
    Vt : np.ndarray
        Transposed right singular vectors of shape (K_max, N).
    k : int
        Number of singular values to retain (1 <= k <= K_max).

    Returns
    -------
    A_k : np.ndarray
        Reconstructed rank-k matrix of shape (M, N).
    """
    k_max = len(s)
    if k < 1 or k > k_max:
        raise ValueError(f"Rank k must satisfy 1 <= k <= {k_max}, got {k}")

    # Efficient vector scaling: (U[:, :k] * s[:k]) @ Vt[:k, :]
    # Avoids constructing a large k x k diagonal matrix
    scaled_U = U[:, :k] * s[:k]
    A_k = np.matmul(scaled_U, Vt[:k, :])
    return A_k


def calculate_energy_retention(s: np.ndarray, k: int) -> float:
    """
    Calculate the percentage of Frobenius energy (variance) retained by rank k.
    Energy(k) = (sum_{i=1}^k sigma_i^2 / sum_{i=1}^{K_max} sigma_i^2) * 100%
    """
    total_energy = np.sum(s ** 2)
    if total_energy == 0.0:
        return 100.0
    retained_energy = np.sum(s[:k] ** 2)
    return float((retained_energy / total_energy) * 100.0)


def svd_storage_analysis(M: int, N: int, k: int) -> Dict[str, Any]:
    """
    Calculate theoretical storage requirements and compression ratio for rank-k SVD.

    Parameters
    ----------
    M : int
        Image height (rows).
    N : int
        Image width (columns).
    k : int
        Approximation rank.

    Returns
    -------
    dict containing:
        - original_elements: M * N
        - svd_elements: k * (M + N + 1)
        - compression_ratio: original / svd
        - breakeven_rank: k* threshold where storage ratio is 1.0
        - is_compressed: True if compression_ratio > 1.0
    """
    orig_elements = M * N
    svd_elements = k * (M + N + 1)
    cr = float(orig_elements / svd_elements) if svd_elements > 0 else 0.0
    breakeven = int((M * N) / (M + N + 1))

    return {
        "original_elements": orig_elements,
        "svd_elements": svd_elements,
        "compression_ratio": cr,
        "breakeven_rank": breakeven,
        "is_compressed": cr > 1.0,
        "u_elements": M * k,
        "sigma_elements": k,
        "vt_elements": k * N
    }


def compress_svd_image(
    image_matrix: np.ndarray,
    k: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform low-rank SVD compression and reconstruction on 2D (grayscale) or 3D (RGB) image.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image in float or uint8, shape (M, N) or (M, N, C).
    k : int
        Approximation rank.

    Returns
    -------
    reconstructed : np.ndarray
        Reconstructed image in float64 [0.0, 255.0].
    metadata : dict
        Computational timing, singular values, energy retained, and storage metrics.
    """
    t_start = time.perf_counter()
    M, N = image_matrix.shape[:2]
    k_max = min(M, N)
    k = max(1, min(k, k_max))

    if image_matrix.ndim == 2:
        # Grayscale channel
        U, s, Vt = compute_svd_thin(image_matrix.astype(np.float64))
        t_decomp = time.perf_counter()
        
        reconstructed = reconstruct_svd_rank_k(U, s, Vt, k)
        t_end = time.perf_counter()

        energy = calculate_energy_retention(s, k)
        storage = svd_storage_analysis(M, N, k)

        metadata = {
            "rank": k,
            "max_rank": k_max,
            "decomposition_time_sec": t_decomp - t_start,
            "reconstruction_time_sec": t_end - t_decomp,
            "total_time_sec": t_end - t_start,
            "singular_values": s,
            "energy_retained_percent": energy,
            "storage": storage,
            "is_color": False
        }
        return reconstructed, metadata

    elif image_matrix.ndim == 3:
        # Multi-channel color processing
        num_channels = image_matrix.shape[2]
        reconstructed_channels = []
        all_s = []
        total_decomp_time = 0.0
        total_recon_time = 0.0

        for c in range(num_channels):
            tc0 = time.perf_counter()
            U, s, Vt = compute_svd_thin(image_matrix[:, :, c].astype(np.float64))
            tc1 = time.perf_counter()
            A_k = reconstruct_svd_rank_k(U, s, Vt, k)
            tc2 = time.perf_counter()

            total_decomp_time += (tc1 - tc0)
            total_recon_time += (tc2 - tc1)
            reconstructed_channels.append(A_k)
            all_s.append(s)

        reconstructed = np.stack(reconstructed_channels, axis=-1)
        t_end = time.perf_counter()

        # Mean singular value spectrum across color channels
        mean_s = np.mean(np.array(all_s), axis=0)
        energy = calculate_energy_retention(mean_s, k)
        
        # In color, 3 channels each require k * (M + N + 1)
        orig_elements = M * N * num_channels
        svd_elements = num_channels * k * (M + N + 1)
        cr = float(orig_elements / svd_elements) if svd_elements > 0 else 0.0

        storage = {
            "original_elements": orig_elements,
            "svd_elements": svd_elements,
            "compression_ratio": cr,
            "breakeven_rank": int((M * N) / (M + N + 1)),
            "is_compressed": cr > 1.0,
            "u_elements": num_channels * M * k,
            "sigma_elements": num_channels * k,
            "vt_elements": num_channels * k * N
        }

        metadata = {
            "rank": k,
            "max_rank": k_max,
            "decomposition_time_sec": total_decomp_time,
            "reconstruction_time_sec": total_recon_time,
            "total_time_sec": t_end - t_start,
            "singular_values": mean_s,
            "energy_retained_percent": energy,
            "storage": storage,
            "is_color": True
        }
        return reconstructed, metadata

    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {image_matrix.shape}")


def verify_svd_on_small_matrix() -> Dict[str, Any]:
    """
    Mathematical verification on a known 4x4 rank-2 matrix.
    Validates:
    1. A = U * Sigma * V^T
    2. U^T * U = I and V^T * V = I (Orthonormality)
    3. Truncated rank-1 error equals sigma_2^2 (Eckart-Young Theorem)
    """
    # Create rank-2 matrix: outer product of 2 linearly independent vectors
    v1 = np.array([1.0, 2.0, 1.0, 0.0])
    v2 = np.array([0.0, 1.0, 2.0, 1.0])
    u1 = np.array([2.0, 0.0, 1.0, 1.0])
    u2 = np.array([1.0, 1.0, 0.0, 2.0])

    A = 3.0 * np.outer(u1, v1) + 1.5 * np.outer(u2, v2)
    U, s, Vt = compute_svd_thin(A)

    # 1. Full reconstruction error
    A_reconstructed = reconstruct_svd_rank_k(U, s, Vt, k=len(s))
    exact_error = float(np.linalg.norm(A - A_reconstructed, "fro"))

    # 2. Orthonormality checks
    ortho_U_error = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1]), "fro"))
    ortho_V_error = float(np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0]), "fro"))

    # 3. Eckart-Young validation: for rank 1, ||A - A_1||_F^2 must equal sum_{i=2}^4 sigma_i^2
    A_1 = reconstruct_svd_rank_k(U, s, Vt, k=1)
    ey_actual_sq_error = float(np.linalg.norm(A - A_1, "fro") ** 2)
    ey_theoretical_sq_error = float(np.sum(s[1:] ** 2))
    ey_difference = abs(ey_actual_sq_error - ey_theoretical_sq_error)

    return {
        "matrix_shape": A.shape,
        "singular_values": s.tolist(),
        "reconstruction_frobenius_error": exact_error,
        "u_orthonormality_error": ortho_U_error,
        "v_orthonormality_error": ortho_V_error,
        "eckart_young_actual_error_sq": ey_actual_sq_error,
        "eckart_young_theoretical_error_sq": ey_theoretical_sq_error,
        "eckart_young_verified": ey_difference < 1e-10
    }
