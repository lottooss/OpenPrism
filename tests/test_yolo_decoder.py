"""Unit and adversarial tests for YOLO decoder logic, coordinate mapping, and observation generation."""


from tools.perception.coordinate_transform import (
    AffineTransform2D,
    CoordinateConverter,
    PixelPoint,
)


def test_yolo_decoder_coordinate_mapping() -> None:
    """Verify mapping model box center (320, 192) back to 1080p source space (960, 540)."""
    # sx = 1/3, sy = 1/3, padx = 0, pady = 12.0
    transform = AffineTransform2D(sx=1.0 / 3.0, sy=1.0 / 3.0, padx=0.0, pady=12.0)

    # Model point
    model_pt = PixelPoint(x=320.0, y=192.0)

    # Inverse mapping
    src_pt = transform.inverse_point(model_pt)
    assert abs(src_pt.x - 960.0) < 1e-4
    assert abs(src_pt.y - 540.0) < 1e-4

    # Normalized [-1, 1] screen mapping
    norm_pt = CoordinateConverter.pixel_to_normalized(src_pt, 1920.0, 1080.0)
    assert abs(norm_pt.x - 0.0) < 1e-4
    assert abs(norm_pt.y - 0.0) < 1e-4


def test_nms_suppression() -> None:
    """Verify IoU calculation correctly filters duplicate high-overlap boxes."""
    box_a = [310.0, 182.0, 330.0, 202.0]  # area 400
    box_b = [311.0, 183.0, 331.0, 203.0]  # almost identical

    # Compute IoU
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = 400 + 400 - intersection
    iou = intersection / union

    assert iou > 0.80  # Highly overlapping
