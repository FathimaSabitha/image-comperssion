"""
Unit Tests for SVD Compression Engine
=====================================
Validates mathematical correctness, Eckart-Young optimality,
orthogonality of singular vectors, and color tensor handling.
"""

import unittest
import numpy as np
from algorithms.svd_engine import (
    compute_svd_thin,
    reconstruct_svd_rank_k,
    calculate_energy_retention,
    svd_storage_analysis,
    compress_svd_image,
    verify_svd_on_small_matrix
)


class TestSVDEngine(unittest.TestCase):

    def test_small_matrix_analytical_verification(self):
        """Verify Eckart-Young theorem and orthogonality on small matrix."""
        results = verify_svd_on_small_matrix()
        self.assertTrue(results["eckart_young_verified"], "Eckart-Young condition failed")
        self.assertLess(results["reconstruction_frobenius_error"], 1e-10)
        self.assertLess(results["u_orthonormality_error"], 1e-10)
        self.assertLess(results["v_orthonormality_error"], 1e-10)

    def test_monotonic_error_decay(self):
        """As rank k increases, reconstruction error must monotonically decrease."""
        np.random.seed(42)
        # 50x40 random positive matrix
        A = np.random.uniform(10.0, 200.0, size=(50, 40))
        U, s, Vt = compute_svd_thin(A)

        prev_error = float("inf")
        for k in range(1, len(s) + 1, 5):
            A_k = reconstruct_svd_rank_k(U, s, Vt, k)
            err = np.linalg.norm(A - A_k, "fro")
            self.assertLessEqual(err, prev_error + 1e-9)
            prev_error = err

    def test_energy_retention_reaches_100(self):
        """At maximum rank, cumulative energy retention must equal 100%."""
        np.random.seed(42)
        A = np.random.rand(30, 25) * 255.0
        _, s, _ = compute_svd_thin(A)
        energy_max = calculate_energy_retention(s, len(s))
        self.assertAlmostEqual(energy_max, 100.0, places=5)

    def test_color_image_svd(self):
        """Verify multi-channel color tensor SVD reconstruction."""
        np.random.seed(42)
        img_color = np.random.randint(0, 256, size=(32, 24, 3), dtype=np.uint8)
        k = 10
        recon, meta = compress_svd_image(img_color, k=k)
        
        self.assertEqual(recon.shape, img_color.shape)
        self.assertTrue(meta["is_color"])
        self.assertEqual(meta["rank"], k)
        self.assertGreater(meta["energy_retained_percent"], 50.0)

    def test_storage_breakeven_threshold(self):
        """Verify the theoretical storage breakeven formula."""
        M, N = 100, 100
        analysis = svd_storage_analysis(M, N, k=20)
        # Breakeven k* = 10000 / 201 ≈ 49
        self.assertEqual(analysis["breakeven_rank"], 49)
        self.assertTrue(analysis["is_compressed"])

        analysis_over = svd_storage_analysis(M, N, k=60)
        self.assertFalse(analysis_over["is_compressed"])
        self.assertLess(analysis_over["compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
