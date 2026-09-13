"""Unit and property tests for centralized coordinate mapping and subpixel round-trips.

Milestone M2-05 verification suite.
"""

from __future__ import annotations

import unittest
import numpy as np

from tools.perception.coordinate_transform import (
    AffineTransform2D,
    BoundingBox,
    CoordinateConverter,
    Covariance2D,
    PixelAcceleration,
    PixelPoint,
    PixelVelocity,
)


class TestCoordinateMapping(unittest.TestCase):
    """Test suite for AffineTransform2D, CoordinateConverter, and subpixel round-trips."""

    def test_standard_resolutions_letterboxing(self) -> None:
        """Verify scaling and padding across common gaming and display resolutions."""
        cases = [
            # (src_w, src_h, dst_w, dst_h, expected_s, expected_padx, expected_pady)
            (1920, 1080, 640, 384, 1.0 / 3.0, 0.0, 12.0),
            (2560, 1440, 640, 384, 0.25, 0.0, 12.0),
            (3840, 2160, 640, 384, 1.0 / 6.0, 0.0, 12.0),
            (1280, 720, 640, 384, 0.5, 0.0, 12.0),
            # 16:10 aspect ratio -> horizontal padding (letterbox columns)
            (1920, 1200, 640, 384, 0.32, 12.8, 0.0),
            # 4:3 aspect ratio -> horizontal padding (pillarbox)
            (800, 600, 640, 384, 0.64, 64.0, 0.0),
            # 1:1 aspect ratio
            (1000, 1000, 640, 384, 0.384, 128.0, 0.0),
        ]

        for src_w, src_h, dst_w, dst_h, exp_s, exp_px, exp_py in cases:
            tf = AffineTransform2D.create_letterbox(src_w, src_h, dst_w, dst_h)
            self.assertAlmostEqual(tf.sx, exp_s, delta=1e-5)
            self.assertAlmostEqual(tf.sy, exp_s, delta=1e-5)
            self.assertAlmostEqual(tf.padx, exp_px, delta=1e-5)
            self.assertAlmostEqual(tf.pady, exp_py, delta=1e-5)

    def test_subpixel_point_roundtrips_across_geometries(self) -> None:
        """Assert round-trip center error <= 0.001 px for arbitrary geometries and edge points."""
        resolutions = [
            (1920, 1080, 640, 384),
            (2560, 1440, 640, 384),
            (3840, 2160, 640, 384),
            (1920, 1200, 640, 384),
            (3440, 1440, 640, 384),
            (800, 600, 640, 384),
        ]

        for src_w, src_h, dst_w, dst_h in resolutions:
            tf = AffineTransform2D.create_letterbox(src_w, src_h, dst_w, dst_h)
            test_pts = [
                PixelPoint(0.0, 0.0),
                PixelPoint(float(src_w), float(src_h)),
                PixelPoint(src_w * 0.5, src_h * 0.5),
                PixelPoint(0.12345, 0.6789),
                PixelPoint(src_w - 0.5, src_h - 0.5),
                PixelPoint(123.4567, 789.0123),
            ]

            for pt in test_pts:
                fwd = tf.forward_point(pt)
                inv = tf.inverse_point(fwd)
                self.assertAlmostEqual(pt.x, inv.x, delta=1e-3)
                self.assertAlmostEqual(pt.y, inv.y, delta=1e-3)

    def test_bounding_box_and_center_roundtrips(self) -> None:
        """Verify bounding box and center coordinate subpixel round-trips."""
        tf = AffineTransform2D.create_letterbox(1920, 1080, 640, 384)
        boxes = [
            BoundingBox(100.25, 200.5, 300.75, 400.25),
            BoundingBox(0.0, 0.0, 1920.0, 1080.0),
            BoundingBox(950.123, 530.456, 969.876, 549.543),
        ]

        for box in boxes:
            box_fwd = tf.forward_box(box)
            box_inv = tf.inverse_box(box_fwd)

            self.assertAlmostEqual(box.left, box_inv.left, delta=1e-3)
            self.assertAlmostEqual(box.top, box_inv.top, delta=1e-3)
            self.assertAlmostEqual(box.right, box_inv.right, delta=1e-3)
            self.assertAlmostEqual(box.bottom, box_inv.bottom, delta=1e-3)

            # Center round-trip
            c_orig = box.center
            c_fwd = box_fwd.center
            c_inv = tf.inverse_point(c_fwd)
            self.assertAlmostEqual(c_orig.x, c_inv.x, delta=1e-3)
            self.assertAlmostEqual(c_orig.y, c_inv.y, delta=1e-3)

    def test_covariance_matrix_scaling_property(self) -> None:
        """Verify covariance transformation Sigma' = A Sigma A^T and exact invertibility."""
        tf = AffineTransform2D.create_letterbox(1920, 1080, 640, 384)
        cov = Covariance2D(xx=16.0, xy=2.5, yy=9.0)

        cov_fwd = tf.forward_covariance(cov)
        cov_inv = tf.inverse_covariance(cov_fwd)

        # Expected forward covariance: (1/3)^2 = 1/9 scaling
        self.assertAlmostEqual(cov_fwd.xx, 16.0 / 9.0, delta=1e-5)
        self.assertAlmostEqual(cov_fwd.xy, 2.5 / 9.0, delta=1e-5)
        self.assertAlmostEqual(cov_fwd.yy, 9.0 / 9.0, delta=1e-5)

        # Inverted covariance bitwise/analytical parity
        self.assertAlmostEqual(cov.xx, cov_inv.xx, delta=1e-3)
        self.assertAlmostEqual(cov.xy, cov_inv.xy, delta=1e-3)
        self.assertAlmostEqual(cov.yy, cov_inv.yy, delta=1e-3)

    def test_kinematics_velocity_and_acceleration(self) -> None:
        """Verify velocity and acceleration transformation properties."""
        tf = AffineTransform2D.create_letterbox(1920, 1080, 640, 384)

        vel = PixelVelocity(x_per_s=300.0, y_per_s=-150.0)
        vel_fwd = tf.forward_velocity(vel)
        vel_inv = tf.inverse_velocity(vel_fwd)

        self.assertAlmostEqual(vel_fwd.x_per_s, 100.0, delta=1e-5)
        self.assertAlmostEqual(vel_fwd.y_per_s, -50.0, delta=1e-5)
        self.assertAlmostEqual(vel.x_per_s, vel_inv.x_per_s, delta=1e-3)
        self.assertAlmostEqual(vel.y_per_s, vel_inv.y_per_s, delta=1e-3)

        acc = PixelAcceleration(x_per_s2=-60.0, y_per_s2=120.0)
        acc_fwd = tf.forward_acceleration(acc)
        acc_inv = tf.inverse_acceleration(acc_fwd)

        self.assertAlmostEqual(acc_fwd.x_per_s2, -20.0, delta=1e-5)
        self.assertAlmostEqual(acc_fwd.y_per_s2, 40.0, delta=1e-5)
        self.assertAlmostEqual(acc.x_per_s2, acc_inv.x_per_s2, delta=1e-3)
        self.assertAlmostEqual(acc.y_per_s2, acc_inv.y_per_s2, delta=1e-3)

    def test_normalized_coordinate_converter(self) -> None:
        """Verify [-1, 1] normalized screen space conversions."""
        w, h = 1920.0, 1080.0
        tf = AffineTransform2D.create_letterbox(1920, 1080, 640, 384)

        # Center point -> (0, 0)
        pt_center = PixelPoint(960.0, 540.0)
        norm_center = CoordinateConverter.pixel_to_normalized(pt_center, w, h)
        self.assertAlmostEqual(norm_center.x, 0.0, delta=1e-6)
        self.assertAlmostEqual(norm_center.y, 0.0, delta=1e-6)

        # Top-left -> (-1, -1)
        pt_tl = PixelPoint(0.0, 0.0)
        norm_tl = CoordinateConverter.pixel_to_normalized(pt_tl, w, h)
        self.assertAlmostEqual(norm_tl.x, -1.0, delta=1e-6)
        self.assertAlmostEqual(norm_tl.y, -1.0, delta=1e-6)

        # Bottom-right -> (+1, +1)
        pt_br = PixelPoint(1920.0, 1080.0)
        norm_br = CoordinateConverter.pixel_to_normalized(pt_br, w, h)
        self.assertAlmostEqual(norm_br.x, 1.0, delta=1e-6)
        self.assertAlmostEqual(norm_br.y, 1.0, delta=1e-6)

        # Round trips from model space to normalized screen space
        model_pts = [
            PixelPoint(320.0, 192.0), # Model center
            PixelPoint(0.0, 12.0),     # Model content top-left
            PixelPoint(640.0, 372.0),  # Model content bottom-right
        ]

        for mpt in model_pts:
            norm_pt = CoordinateConverter.model_to_normalized_screen(mpt, tf, w, h)
            mpt_back = CoordinateConverter.normalized_screen_to_model(norm_pt, tf, w, h)
            self.assertAlmostEqual(mpt.x, mpt_back.x, delta=1e-3)
            self.assertAlmostEqual(mpt.y, mpt_back.y, delta=1e-3)

    def test_affine_matrices_inversion(self) -> None:
        """Assert M_inv * M_fwd == I for both 2x3 and 3x3 matrices."""
        tf = AffineTransform2D.create_letterbox(1920, 1080, 640, 384)
        m_fwd = tf.forward_matrix_3x3()
        m_inv = tf.inverse_matrix_3x3()

        identity_mat = np.matmul(m_inv, m_fwd)
        np.testing.assert_allclose(identity_mat, np.eye(3), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
