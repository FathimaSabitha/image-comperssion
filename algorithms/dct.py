"""
DCT Algorithm Wrapper
Re-exports from algorithms.dct_engine
"""
from algorithms.dct_engine import (
    JPEG_LUMINANCE_QUANT_TABLE,
    JPEG_CHROMINANCE_QUANT_TABLE,
    DCT_MATRIX_8X8,
    generate_dct_basis_matrix,
    scale_quantization_matrix,
    forward_dct_block,
    inverse_dct_block,
    quantize_block,
    dequantize_block,
    zigzag_scan,
    calculate_entropy,
    compress_dct_channel_2d,
    compress_dct_image,
    verify_dct_on_small_block
)
