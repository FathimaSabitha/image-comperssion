"""
Unit Tests for DCT Compression Engine
=====================================
Validates 2D DCT-II mathematical orthonormality, invertibility,
quantization matrix scaling, and block boundary padding.
"""

import unittest
import numpy as np
from algorithms.dct_engine import (
    generate_dct_basis_matrix,
    scale_quantization_matrix,
    JPEG_LUMINANCE_QUANT_TABLE,
    forward_dct_block,
    inverse_dct_block,
    compress_dct_image,
    verify_dct_on_small_block
)
from processing.image_loader import pad_to_block_size, crop_to_original_size


class TestDCTEngine(unittest.TestCase):

    def test_orthonormality_and_invertibility(self):
        """Verify mathematical orthonormality and exact inversion on small block."""
        results = verify_dct_on_small_block()
        self.assertTrue(results["is_orthonormal"], "DCT matrix is not orthonormal")
        self.assertTrue(results["is_invertible"], "DCT exact invertibility failed")
        self.assertLess(results["exact_inversion_error"], 1e-11)

    def test_quantization_table_scaling(self):
        """Verify quality scaling factor monotonically reduces quantization step."""
        base_Q = JPEG_LUMINANCE_QUANT_TABLE
        Q_low = scale_quantization_matrix(base_Q, quality=10)
        Q_mid = scale_quantization_matrix(base_Q, quality=50)
        Q_high = scale_quantization_matrix(base_Q, quality=90)

        # Higher quality factor => smaller quantization steps
        self.assertTrue(np.all(Q_low >= Q_mid))
        self.assertTrue(np.all(Q_mid >= Q_high))
        self.assertTrue(np.all(Q_high >= 1.0))

    def test_padding_and_unpadding(self):
        """Verify that arbitrary dimension images are padded to multiples of 8 and restored."""
        orig_shape = (73, 51)
        matrix = np.random.randint(0, 255, size=orig_shape, dtype=np.uint8)
        
        padded, pad_amt = pad_to_block_size(matrix, block_size=8)
        self.assertEqual(padded.shape[0] % 8, 0)
        self.assertEqual(padded.shape[1] % 8, 0)
        
        cropped = crop_to_original_size(padded, orig_shape)
        self.assertEqual(cropped.shape, orig_shape)
        np.testing.assert_array_equal(cropped, matrix)

    def test_sparsity_monotonicity_with_quality(self):
        """Lower quality factors should induce higher coefficient sparsity (more zeros)."""
        np.random.seed(101)
        test_img = np.random.randint(20, 220, size=(64, 64), dtype=np.uint8)

        _, meta_q10 = compress_dct_image(test_img, quality=10)
        _, meta_q90 = compress_dct_image(test_img, quality=90)

        self.assertGreater(meta_q10["sparsity_percent"], meta_q90["sparsity_percent"])
        self.assertGreater(meta_q10["theoretical_cr_entropy"], meta_q90["theoretical_cr_entropy"])


if __name__ == "__main__":
    unittest.main()
