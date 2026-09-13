from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class MouseButton(IntEnum):
  None_ = cast(int, ...)
  Left = cast(int, ...)
  Right = cast(int, ...)
  Middle = cast(int, ...)
  X1 = cast(int, ...)
  X2 = cast(int, ...)
