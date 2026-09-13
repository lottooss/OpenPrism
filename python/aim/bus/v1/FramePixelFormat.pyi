from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class FramePixelFormat(IntEnum):
  Unknown = cast(int, ...)
  B8G8R8A8_UNORM = cast(int, ...)
  R8G8B8A8_UNORM = cast(int, ...)
  R16G16B16A16_FLOAT = cast(int, ...)
  NV12 = cast(int, ...)
