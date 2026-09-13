from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class Visibility(IntEnum):
  Visible = cast(int, ...)
  Partial = cast(int, ...)
  Predicted = cast(int, ...)
