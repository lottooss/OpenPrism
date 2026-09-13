"""Monotonic Clock abstractions and QPC-to-nanosecond conversion."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
from typing import Protocol


def qpc_to_ns_euclidean(qpc: int, qpf: int) -> int:
    """Convert QPC ticks to nanoseconds using overflow-safe Euclidean division.

    Separates whole seconds from fractional remainder:
        T_ns = (Q * 1_000_000_000) + (R * 1_000_000_000 // QPF)
    where Q = qpc // qpf, R = qpc % qpf.

    Guaranteed overflow-free for over 584 years of continuous uptime on 64-bit systems.
    """
    if qpf == 0:
        return 0
    if qpf == 10_000_000:
        return qpc * 100
    q = qpc // qpf
    r = qpc % qpf
    return (q * 1_000_000_000) + ((r * 1_000_000_000) // qpf)


class Clock(Protocol):
    """Protocol for monotonic nanosecond clocks."""
    def now_ns(self) -> int:
        ...


class QpcClock:
    """Windows QueryPerformanceCounter hardware clock with monotonic clamp."""

    def __init__(self) -> None:
        self._kernel32 = None
        self._qpf = 10_000_000
        self._is_10mhz = True
        self._last_ns = 0

        try:
            self._kernel32 = ctypes.windll.kernel32
            freq = wintypes.LARGE_INTEGER()
            if self._kernel32.QueryPerformanceFrequency(ctypes.byref(freq)):
                self._qpf = freq.value
                self._is_10mhz = (self._qpf == 10_000_000)
        except Exception:
            pass

    @property
    def qpf(self) -> int:
        return self._qpf

    @property
    def is_10mhz(self) -> bool:
        return self._is_10mhz

    def now_ns(self) -> int:
        if self._kernel32 is not None:
            count = wintypes.LARGE_INTEGER()
            if self._kernel32.QueryPerformanceCounter(ctypes.byref(count)):
                raw_qpc = count.value
                if self._is_10mhz:
                    sample_ns = raw_qpc * 100
                else:
                    sample_ns = qpc_to_ns_euclidean(raw_qpc, self._qpf)
            else:
                sample_ns = time.perf_counter_ns()
        else:
            sample_ns = time.perf_counter_ns()

        # Monotonic non-decreasing clamp
        if sample_ns < self._last_ns:
            sample_ns = self._last_ns
        else:
            self._last_ns = sample_ns

        return sample_ns


class FakeClock:
    """Deterministic stepped mock clock for reproducible testing and replays."""

    def __init__(self, initial_ns: int = 0) -> None:
        self._current_ns = initial_ns

    def now_ns(self) -> int:
        return self._current_ns

    def advance_ns(self, delta_ns: int) -> None:
        if delta_ns < 0:
            raise ValueError("Cannot advance clock backwards in monotonic time")
        self._current_ns += delta_ns

    def advance_ms(self, delta_ms: float) -> None:
        self.advance_ns(int(delta_ms * 1_000_000))

    def set_ns(self, target_ns: int) -> None:
        if target_ns < self._current_ns:
            raise ValueError("Cannot set clock to a time in the past")
        self._current_ns = target_ns
