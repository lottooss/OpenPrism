from __future__ import annotations

import flatbuffers
import numpy as np

import typing
from enum import IntEnum
from typing import cast

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class AimMode(IntEnum):
  Idle = cast(int, ...)
  Tracking = cast(int, ...)
  Flick = cast(int, ...)
  MicroCorrection = cast(int, ...)
  EmergencyHold = cast(int, ...)
