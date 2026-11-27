"""
Preprocessing Module
====================
Mathematical preprocessing for image matrices:
- Precision management (uint8 <-> float64)
- Color space transformations (RGB <-> Grayscale, RGB <-> YCbCr)
- Spatial block partitioning and reconstruction (8x8 blocks)
"""

from typing import List, Tuple
import numpy as np


def to_float_matrix(img_array: np.ndarray) -> np.ndarray:
    """
    Convert integer pixel intensities in [0, 255] to float64 representation.
    Preserves dynamic range [0.0, 255.0] for exact mathematical decomposition.

    Parameters
    ----------
    img_array : np.ndarray
        Array of integer pixel values.

    Returns
    -------
    np.ndarray
        Float64 array of identical shape.
    """
    return np.asarray(img_array, dtype=np.float64)


def to_uint8_image(matrix: np.ndarray) -> np.ndarray:
    """
    Clip and round floating-point reconstructed pixel values back to [0, 255] uint8.
    Ensures mathematical reconstruction overshoots/undershoots are bounded.

    Parameters
    ----------
    matrix : np.ndarray
        Floating-point reconstructed pixel matrix.

    Returns
    -------
    np.ndarray
        uint8 image array in range [0, 255].
    """
    clipped = np.clip(matrix, 0.0, 255.0)
    return np.round(clipped).astype(np.uint8)


def rgb_to_grayscale(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to Grayscale using standard ITU-R BT.601 luma weights:
    Y = 0.299*R + 0.587*G + 0.114*B

    Parameters
    ----------
    rgb_array : np.ndarray
        RGB image of shape (M, N, 3).

    Returns
    -------
    np.ndarray
        Grayscale matrix of shape (M, N) with dtype uint8.
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected shape (M, N, 3), got {rgb_array.shape}")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = np.tensordot(rgb_array.astype(np.float64), weights, axes=(-1, -1))
    return to_uint8_image(gray)


def rgb_to_ycbcr(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB color image to YCbCr color space (ITU-R BT.601).
    Used in standard JPEG compression to decorrelate luminance from chrominance.

    Parameters
    ----------
    rgb_array : np.ndarray
        RGB image of shape (M, N, 3) in [0, 255].

    Returns
    -------
    np.ndarray
        YCbCr array of shape (M, N, 3) in float64.
        Y in [0, 255], Cb in [0, 255], Cr in [0, 255].
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected shape (M, N, 3), got {rgb_array.shape}")

    rgb = rgb_array.astype(np.float64)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycbcr_array: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr color space back to RGB (ITU-R BT.601).

    Parameters
    ----------
    ycbcr_array : np.ndarray
        YCbCr image of shape (M, N, 3) in float64.

    Returns
    -------
    np.ndarray
        RGB image of shape (M, N, 3) in uint8 [0, 255].
    """
    if ycbcr_array.ndim != 3 or ycbcr_array.shape[2] != 3:
        raise ValueError(f"Expected shape (M, N, 3), got {ycbcr_array.shape}")

    y = ycbcr_array[:, :, 0]
    cb = ycbcr_array[:, :, 1] - 128.0
    cr = ycbcr_array[:, :, 2] - 128.0

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    rgb = np.stack([r, g, b], axis=-1)
    return to_uint8_image(rgb)


def split_channels(tensor: np.ndarray) -> List[np.ndarray]:
    """
    Split a 3D image tensor (M, N, C) into a list of 2D matrices [(M, N), ...].
    If tensor is already 2D (M, N), returns [tensor].
    """
    if tensor.ndim == 2:
        return [tensor]
    elif tensor.ndim == 3:
        return [tensor[:, :, c] for c in range(tensor.shape[2])]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")


def merge_channels(channels: List[np.ndarray]) -> np.ndarray:
    """
    Merge a list of 2D channel matrices back into a 3D tensor or single 2D matrix.
    """
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=-1)


def block_partition_2d(matrix_2d: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Partition a 2D matrix (M, N) into non-overlapping blocks of size (B, B).
    Assumes M and N are exact multiples of block_size.

    Parameters
    ----------
    matrix_2d : np.ndarray
        2D matrix of shape (M, N).
    block_size : int, default=8
        Block size B.

    Returns
    -------
    blocks : np.ndarray
        4D array of shape (M // B, N // B, B, B).
    """
    m, n = matrix_2d.shape
    if m % block_size != 0 or n % block_size != 0:
        raise ValueError(f"Dimensions ({m}, {n}) must be divisible by {block_size}")

    num_h = m // block_size
    num_w = n // block_size

    # Efficient reshape and transpose without copying memory:
    # (num_h, block_size, num_w, block_size) -> (num_h, num_w, block_size, block_size)
    blocks = (
        matrix_2d.reshape(num_h, block_size, num_w, block_size)
        .swapaxes(1, 2)
    )
    return blocks


def block_stitch_2d(blocks_4d: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 2D matrix (M, N) from a 4D array of blocks (num_h, num_w, B, B).

    Parameters
    ----------
    blocks_4d : np.ndarray
        4D array of shape (num_h, num_w, B, B).

    Returns
    -------
    matrix_2d : np.ndarray
        Reconstructed 2D matrix of shape (num_h * B, num_w * B).
    """
    num_h, num_w, b_h, b_w = blocks_4d.shape
    return (
        blocks_4d.swapaxes(1, 2)
        .reshape(num_h * b_h, num_w * b_w)
    )
