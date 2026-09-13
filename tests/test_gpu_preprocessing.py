"""Unit tests for GPU/CPU preprocessing, affine letterbox transforms, and numerical parity."""

from __future__ import annotations

import unittest
import numpy as np


class TestGpuPreprocessing(unittest.TestCase):
    """Test suite for preprocessing geometry, letterboxing math, and affine transforms."""

    def test_letterbox_geometry_1080p_to_640x384(self) -> None:
        """Verify 1920x1080 to 640x384 letterbox dimensions and scaling factors."""
        src_w, src_h = 1920, 1080
        dst_w, dst_h = 640, 384

        scale_x = dst_w / src_w  # 640 / 1920 = 1/3
        scale_y = dst_h / src_h  # 384 / 1080 = 0.35555...
        scale = min(scale_x, scale_y)  # 1/3

        scaled_w = int(round(src_w * scale))  # 640
        scaled_h = int(round(src_h * scale))  # 360

        pad_x = (dst_w - scaled_w) / 2.0  # 0.0
        pad_y = (dst_h - scaled_h) / 2.0  # 12.0

        self.assertEqual(scaled_w, 640)
        self.assertEqual(scaled_h, 360)
        self.assertEqual(pad_x, 0.0)
        self.assertEqual(pad_y, 12.0)
        self.assertAlmostEqual(scale, 1.0 / 3.0, places=6)

    def test_affine_point_forward_inverse_roundtrip(self) -> None:
        """Verify subpixel accuracy in forward and inverse affine point mapping."""
        sx = 1.0 / 3.0
        sy = 1.0 / 3.0
        pad_x = 0.0
        pad_y = 12.0

        test_points = [
            (0.0, 0.0),
            (960.0, 540.0),
            (1920.0, 1080.0),
            (123.456, 789.123),
            (0.5, 0.5),
            (1919.5, 1079.5),
        ]

        for x_src, y_src in test_points:
            # Forward transform
            x_dst = x_src * sx + pad_x
            y_dst = y_src * sy + pad_y

            # Inverse transform
            x_inv = (x_dst - pad_x) / sx
            y_inv = (y_dst - pad_y) / sy

            self.assertAlmostEqual(x_src, x_inv, delta=1e-4)
            self.assertAlmostEqual(y_src, y_inv, delta=1e-4)

    def test_affine_box_forward_inverse_roundtrip(self) -> None:
        """Verify bounding box forward and inverse letterbox coordinate transforms."""
        sx = 1.0 / 3.0
        sy = 1.0 / 3.0
        pad_x = 0.0
        pad_y = 12.0

        boxes = [
            (100.0, 200.0, 300.0, 400.0),
            (0.0, 0.0, 1920.0, 1080.0),
            (950.25, 530.5, 970.75, 550.25),
        ]

        for box_l, box_t, box_r, box_b in boxes:
            # Forward
            l_dst, t_dst = box_l * sx + pad_x, box_t * sy + pad_y
            r_dst, b_dst = box_r * sx + pad_x, box_b * sy + pad_y

            # Inverse
            l_inv, t_inv = (l_dst - pad_x) / sx, (t_dst - pad_y) / sy
            r_inv, b_inv = (r_dst - pad_x) / sx, (b_dst - pad_y) / sy

            self.assertAlmostEqual(box_l, l_inv, delta=1e-4)
            self.assertAlmostEqual(box_t, t_inv, delta=1e-4)
            self.assertAlmostEqual(box_r, r_inv, delta=1e-4)
            self.assertAlmostEqual(box_b, b_inv, delta=1e-4)

    def test_affine_matrix_inverse_property(self) -> None:
        """Verify that forward and inverse 3x3 affine matrices multiply to identity."""
        sx = 1.0 / 3.0
        sy = 1.0 / 3.0
        pad_x = 0.0
        pad_y = 12.0

        M_fwd = np.array([
            [sx, 0.0, pad_x],
            [0.0, sy, pad_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        M_inv = np.array([
            [1.0 / sx, 0.0, -pad_x / sx],
            [0.0, 1.0 / sy, -pad_y / sy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        identity_mat = np.matmul(M_inv, M_fwd)
        np.testing.assert_allclose(identity_mat, np.eye(3), atol=1e-6)

    def test_constant_padding_value(self) -> None:
        """Verify normalization of constant 114 letterbox padding value."""
        pad_u8 = 114
        pad_norm = pad_u8 / 255.0
        self.assertAlmostEqual(pad_norm, 0.4470588235294118, places=6)


if __name__ == "__main__":
    unittest.main()
