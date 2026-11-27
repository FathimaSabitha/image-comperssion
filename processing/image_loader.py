"""
Image Loader Module
===================
Handles robust loading, validation, format conversion, and spatial padding
of digital images for mathematical matrix processing.
"""

from typing import Union, Tuple, Optional
import io
import os
import numpy as np
from PIL import Image


def load_image(
    source: Union[str, bytes, io.BytesIO, Image.Image],
    as_grayscale: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load an image from a file path, byte stream, or PIL Image object.

    Parameters
    ----------
    source : str, bytes, BytesIO, or PIL.Image.Image
        Input image source.
    as_grayscale : bool, default=True
        If True, converts image to single-channel 8-bit grayscale (L).
        If False, converts to standard 3-channel RGB.

    Returns
    -------
    img_array : np.ndarray
        NumPy array of dtype uint8 representing pixel intensities in [0, 255].
        Shape is (M, N) for grayscale or (M, N, 3) for RGB.
    original_dims : tuple of (int, int)
        (height, width) of the original unpadded image.
    """
    if isinstance(source, Image.Image):
        pil_img = source
    elif isinstance(source, (str, os.PathLike)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        pil_img = Image.open(source)
    elif isinstance(source, (bytes, io.BytesIO)):
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        pil_img = Image.open(source)
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    if as_grayscale:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    img_array = np.array(pil_img, dtype=np.uint8)
    original_dims = (img_array.shape[0], img_array.shape[1])
    return img_array, original_dims


def pad_to_block_size(
    img_array: np.ndarray,
    block_size: int = 8,
    mode: str = "edge"
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad an image matrix so its dimensions are exact multiples of block_size.
    Required for block-based transforms such as 8x8 DCT.

    Parameters
    ----------
    img_array : np.ndarray
        Input 2D (M, N) or 3D (M, N, C) image array.
    block_size : int, default=8
        Block dimension (e.g. 8 for 8x8 DCT).
    mode : str, default='edge'
        Padding mode supported by np.pad ('edge', 'reflect', or 'constant').

    Returns
    -------
    padded_array : np.ndarray
        Array padded such that padded_array.shape[0] % block_size == 0
        and padded_array.shape[1] % block_size == 0.
    padding_amounts : tuple of (pad_h, pad_w)
        Number of rows and columns appended.
    """
    h, w = img_array.shape[:2]
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h == 0 and pad_w == 0:
        return img_array.copy(), (0, 0)

    if img_array.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    elif img_array.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {img_array.shape}")

    padded_array = np.pad(img_array, pad_width=pad_width, mode=mode)
    return padded_array, (pad_h, pad_w)


def crop_to_original_size(
    padded_array: np.ndarray,
    original_dims: Tuple[int, int]
) -> np.ndarray:
    """
    Crop a padded matrix back to its original (height, width).

    Parameters
    ----------
    padded_array : np.ndarray
        Padded 2D or 3D image array.
    original_dims : tuple of (int, int)
        Original (height, width) before padding.

    Returns
    -------
    cropped_array : np.ndarray
        Array cropped to original_dims.
    """
    orig_h, orig_w = original_dims
    if padded_array.ndim == 2:
        return padded_array[:orig_h, :orig_w].copy()
    elif padded_array.ndim == 3:
        return padded_array[:orig_h, :orig_w, :].copy()
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {padded_array.shape}")
