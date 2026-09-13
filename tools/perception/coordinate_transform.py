"""Canonical 2D affine letterbox transformation and coordinate conversion utilities.

Matches C++20 aim::perception::AffineTransform2D and CoordinateConverter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


class PixelPoint(NamedTuple):
    x: float
    y: float


class NormalizedPoint(NamedTuple):
    x: float
    y: float


class BoundingBox(NamedTuple):
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return abs(self.right - self.left)

    @property
    def height(self) -> float:
        return abs(self.bottom - self.top)

    @property
    def center(self) -> PixelPoint:
        return PixelPoint(
            x=(self.left + self.right) * 0.5,
            y=(self.top + self.bottom) * 0.5,
        )


class PixelVelocity(NamedTuple):
    x_per_s: float
    y_per_s: float


class PixelAcceleration(NamedTuple):
    x_per_s2: float
    y_per_s2: float


class Covariance2D(NamedTuple):
    xx: float
    xy: float
    yy: float


@dataclass(frozen=True)
class AffineTransform2D:
    """Canonical 2D affine letterbox transformation parameters."""

    sx: float = 1.0 / 3.0
    sy: float = 1.0 / 3.0
    padx: float = 0.0
    pady: float = 12.0

    @classmethod
    def create_letterbox(
        cls,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
    ) -> AffineTransform2D:
        """Calculates scale and padding for uniform aspect-ratio letterboxing."""
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return cls(1.0, 1.0, 0.0, 0.0)

        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scale = min(scale_x, scale_y)

        scaled_w = src_w * scale
        scaled_h = src_h * scale

        pad_x = (dst_w - scaled_w) * 0.5
        pad_y = (dst_h - scaled_h) * 0.5

        return cls(sx=scale, sy=scale, padx=pad_x, pady=pad_y)

    def forward_matrix_2x3(self) -> np.ndarray:
        """Returns 2x3 forward transformation matrix."""
        return np.array([
            [self.sx, 0.0, self.padx],
            [0.0, self.sy, self.pady],
        ], dtype=np.float64)

    def forward_matrix_3x3(self) -> np.ndarray:
        """Returns 3x3 forward transformation matrix."""
        return np.array([
            [self.sx, 0.0, self.padx],
            [0.0, self.sy, self.pady],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def inverse_matrix_2x3(self) -> np.ndarray:
        """Returns 2x3 inverse transformation matrix."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return np.array([
            [inv_sx, 0.0, -self.padx * inv_sx],
            [0.0, inv_sy, -self.pady * inv_sy],
        ], dtype=np.float64)

    def inverse_matrix_3x3(self) -> np.ndarray:
        """Returns 3x3 inverse transformation matrix."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return np.array([
            [inv_sx, 0.0, -self.padx * inv_sx],
            [0.0, inv_sy, -self.pady * inv_sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def forward_point(self, pt: PixelPoint) -> PixelPoint:
        """Projects screen pixel point to model tensor space."""
        return PixelPoint(
            x=pt.x * self.sx + self.padx,
            y=pt.y * self.sy + self.pady,
        )

    def inverse_point(self, pt: PixelPoint) -> PixelPoint:
        """Inverts model tensor point back to screen pixel space."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return PixelPoint(
            x=(pt.x - self.padx) * inv_sx,
            y=(pt.y - self.pady) * inv_sy,
        )

    def forward_box(self, box: BoundingBox) -> BoundingBox:
        """Projects screen bounding box to model tensor space."""
        return BoundingBox(
            left=box.left * self.sx + self.padx,
            top=box.top * self.sy + self.pady,
            right=box.right * self.sx + self.padx,
            bottom=box.bottom * self.sy + self.pady,
        )

    def inverse_box(self, box: BoundingBox) -> BoundingBox:
        """Inverts model tensor bounding box back to screen space."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return BoundingBox(
            left=(box.left - self.padx) * inv_sx,
            top=(box.top - self.pady) * inv_sy,
            right=(box.right - self.padx) * inv_sx,
            bottom=(box.bottom - self.pady) * inv_sy,
        )

    def forward_radius(self, r: float) -> float:
        """Scales scalar radius to model tensor space."""
        return r * math.sqrt(self.sx * self.sy)

    def inverse_radius(self, r: float) -> float:
        """Inverts model tensor radius to screen space."""
        scale = math.sqrt(self.sx * self.sy)
        return (r / scale) if scale != 0.0 else 0.0

    def forward_velocity(self, vel: PixelVelocity) -> PixelVelocity:
        """Scales velocity vector (translation invariant)."""
        return PixelVelocity(
            x_per_s=vel.x_per_s * self.sx,
            y_per_s=vel.y_per_s * self.sy,
        )

    def inverse_velocity(self, vel: PixelVelocity) -> PixelVelocity:
        """Inverts velocity vector to screen space."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return PixelVelocity(
            x_per_s=vel.x_per_s * inv_sx,
            y_per_s=vel.y_per_s * inv_sy,
        )

    def forward_acceleration(self, acc: PixelAcceleration) -> PixelAcceleration:
        """Scales acceleration vector (translation invariant)."""
        return PixelAcceleration(
            x_per_s2=acc.x_per_s2 * self.sx,
            y_per_s2=acc.y_per_s2 * self.sy,
        )

    def inverse_acceleration(self, acc: PixelAcceleration) -> PixelAcceleration:
        """Inverts acceleration vector to screen space."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return PixelAcceleration(
            x_per_s2=acc.x_per_s2 * inv_sx,
            y_per_s2=acc.y_per_s2 * inv_sy,
        )

    def forward_covariance(self, cov: Covariance2D) -> Covariance2D:
        """Transforms covariance matrix Sigma' = A * Sigma * A^T."""
        return Covariance2D(
            xx=cov.xx * (self.sx * self.sx),
            xy=cov.xy * (self.sx * self.sy),
            yy=cov.yy * (self.sy * self.sy),
        )

    def inverse_covariance(self, cov: Covariance2D) -> Covariance2D:
        """Inverts covariance matrix to screen space."""
        inv_sx = 1.0 / self.sx if self.sx != 0.0 else 0.0
        inv_sy = 1.0 / self.sy if self.sy != 0.0 else 0.0
        return Covariance2D(
            xx=cov.xx * (inv_sx * inv_sx),
            xy=cov.xy * (inv_sx * inv_sy),
            yy=cov.yy * (inv_sy * inv_sy),
        )


class CoordinateConverter:
    """Utility converting between pixel coordinates and canonical [-1, 1] normalized screen space."""

    @staticmethod
    def pixel_to_normalized(
        pt: PixelPoint,
        screen_w: float,
        screen_h: float,
    ) -> NormalizedPoint:
        """Converts screen pixels to [-1, 1] normalized coordinates.

        (0, 0) is screen center, (-1, -1) is top-left, (+1, +1) is bottom-right.
        """
        if screen_w <= 0.0 or screen_h <= 0.0:
            return NormalizedPoint(0.0, 0.0)
        norm_x = (pt.x / (screen_w * 0.5)) - 1.0
        norm_y = (pt.y / (screen_h * 0.5)) - 1.0
        return NormalizedPoint(norm_x, norm_y)

    @staticmethod
    def normalized_to_pixel(
        npt: NormalizedPoint,
        screen_w: float,
        screen_h: float,
    ) -> PixelPoint:
        """Converts [-1, 1] normalized coordinates to screen pixels."""
        if screen_w <= 0.0 or screen_h <= 0.0:
            return PixelPoint(0.0, 0.0)
        px_x = (npt.x + 1.0) * (screen_w * 0.5)
        px_y = (npt.y + 1.0) * (screen_h * 0.5)
        return PixelPoint(px_x, px_y)

    @staticmethod
    def model_to_normalized_screen(
        model_pt: PixelPoint,
        transform: AffineTransform2D,
        screen_w: float,
        screen_h: float,
    ) -> NormalizedPoint:
        """Directly converts model tensor coordinate to normalized screen [-1, 1]."""
        src_pt = transform.inverse_point(model_pt)
        return CoordinateConverter.pixel_to_normalized(src_pt, screen_w, screen_h)

    @staticmethod
    def normalized_screen_to_model(
        norm_pt: NormalizedPoint,
        transform: AffineTransform2D,
        screen_w: float,
        screen_h: float,
    ) -> PixelPoint:
        """Directly converts normalized screen [-1, 1] to model tensor coordinate."""
        src_pt = CoordinateConverter.normalized_to_pixel(norm_pt, screen_w, screen_h)
        return transform.forward_point(src_pt)
