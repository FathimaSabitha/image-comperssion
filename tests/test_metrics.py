"""
Unit Tests for Evaluation Metrics and Storage Analysis
======================================================
Validates statistical equations (MSE, RMSE, PSNR, SSIM)
and theoretical storage / entropy calculations.
"""

import unittest
import numpy as np
from evaluation.metrics import (
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    compute_all_metrics
)
from evaluation.storage_analysis import analyze_svd_storage, analyze_dct_storage
from algorithms.dct_engine import calculate_entropy


class TestMetrics(unittest.TestCase):

    def test_identical_matrices_perfect_score(self):
        """Identical matrices must have MSE=0, PSNR=100dB, SSIM=1.0."""
        A = np.full((32, 32), 128.0)
        metrics = compute_all_metrics(A, A)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["psnr_db"], 100.0)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=4)

    def test_known_constant_error_psnr(self):
        """Uniform difference of 10 must yield MSE=100 and PSNR=28.1308 dB."""
        A = np.full((20, 20), 100.0)
        B = np.full((20, 20), 110.0)
        mse = calculate_mse(A, B)
        self.assertEqual(mse, 100.0)
        expected_psnr = 10.0 * np.log10((255.0 ** 2) / 100.0)
        psnr = calculate_psnr(A, B)
        self.assertAlmostEqual(psnr, expected_psnr, places=4)

    def test_shannon_entropy_calculation(self):
        """Uniform distribution across 4 symbols has entropy exactly log2(4) = 2.0 bits."""
        # 4 distinct symbols with equal frequency (25 each)
        symbols = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        h = calculate_entropy(symbols)
        self.assertAlmostEqual(h, 2.0, places=5)

    def test_svd_storage_dimensions(self):
        """Test SVD parameter counting formula k*(M + N + 1)."""
        shape = (100, 80)
        k = 10
        analysis = analyze_svd_storage(shape, k=k)
        expected_params = 10 * (100 + 80 + 1)  # 1810
        self.assertEqual(analysis["svd_parameter_count"], expected_params)
        self.assertEqual(analysis["raw_parameter_count"], 8000)
        expected_cr = 8000.0 / 1810.0
        self.assertAlmostEqual(analysis["parameter_compression_ratio"], expected_cr, places=4)


if __name__ == "__main__":
    unittest.main()
