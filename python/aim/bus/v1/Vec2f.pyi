from __future__ import annotations

import flatbuffers
import numpy as np

import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class Vec2f:
  @classmethod
  def SizeOf(cls) -> int: ...

  def Init(self, buf: bytes, pos: int) -> None: ...
  def X(self) -> float: ...
  def Y(self) -> float: ...

def CreateVec2f(builder: flatbuffers.Builder, x: float, y: float) -> uoffset: ...
