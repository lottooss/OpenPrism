from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class CorrelationFlags(IntEnum):
  None_ = cast(int, ...)
  Synthetic = cast(int, ...)
  Warmup = cast(int, ...)
  Dropped = cast(int, ...)
  TraceVerbose = cast(int, ...)
