from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class TrackState(IntEnum):
  Tentative = cast(int, ...)
  Confirmed = cast(int, ...)
  Occluded = cast(int, ...)
  Coasting = cast(int, ...)
  Lost = cast(int, ...)
