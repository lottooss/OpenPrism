from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class CoordinateSpace(IntEnum):
  PixelTopLeft = cast(int, ...)
  NormalizedCenter = cast(int, ...)
  CameraSpherical = cast(int, ...)
