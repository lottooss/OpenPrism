from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class ButtonAction(IntEnum):
  None_ = cast(int, ...)
  Press = cast(int, ...)
  Release = cast(int, ...)
  Click = cast(int, ...)
