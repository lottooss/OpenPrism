"""tools/sim/replay_clock.py
Steppable deterministic replay clock with driftless fractional nanosecond accumulation.
"""

from __future__ import annotations



class ReplayClock:
    """Deterministic simulation clock with exact fractional nanosecond timekeeping."""

    def __init__(self, cadence_hz: float = 144.0, start_time_ns: int = 1_000_000_000) -> None:
        self._cadence_hz: float = cadence_hz if cadence_hz > 0 else 144.0
        self._start_time_ns: int = start_time_ns
        self._current_time_ns: int = start_time_ns
        self._tick_index: int = 0
        self._frame_id: int = 0
        self._sequence_id: int = 0
        self._rate_scale: float = 1.0
        self._is_paused: bool = False

    def now_ns(self) -> int:
        """Returns current monotonic virtual timestamp."""
        return self._current_time_ns

    @property
    def cadence_hz(self) -> float:
        return self._cadence_hz

    @property
    def start_time_ns(self) -> int:
        return self._start_time_ns

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def sequence_id(self) -> int:
        return self._sequence_id

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def rate_scale(self) -> float:
        return self._rate_scale

    @property
    def elapsed_ns(self) -> int:
        return self._current_time_ns - self._start_time_ns

    @property
    def elapsed_seconds(self) -> float:
        return float(self.elapsed_ns) / 1_000_000_000.0

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def set_rate_scale(self, scale: float) -> None:
        self._rate_scale = scale if scale >= 0 else 1.0

    def step(self, count: int = 1) -> None:
        """Advances clock by count ticks using exact fractional calculation."""
        if self._is_paused or count <= 0:
            return

        self._tick_index += count
        self._frame_id += count
        self._sequence_id += count

        total_ns = int((float(self._tick_index) * 1_000_000_000.0 * self._rate_scale) / self._cadence_hz)
        self._current_time_ns = self._start_time_ns + total_ns

    def advance_tick(self) -> None:
        self.step(1)

    def seek_tick(self, target_tick: int) -> None:
        self._tick_index = target_tick
        self._frame_id = target_tick
        self._sequence_id = target_tick
        total_ns = int((float(self._tick_index) * 1_000_000_000.0 * self._rate_scale) / self._cadence_hz)
        self._current_time_ns = self._start_time_ns + total_ns

    def seek_ns(self, target_ns: int) -> None:
        if target_ns < self._start_time_ns:
            target_ns = self._start_time_ns
        delta_sec = float(target_ns - self._start_time_ns) / 1_000_000_000.0
        effective_hz = (self._cadence_hz / self._rate_scale) if self._rate_scale > 0 else self._cadence_hz
        self._tick_index = int(round(delta_sec * effective_hz))
        self._frame_id = self._tick_index
        self._sequence_id = self._tick_index
        total_ns = int((float(self._tick_index) * 1_000_000_000.0 * self._rate_scale) / self._cadence_hz)
        self._current_time_ns = self._start_time_ns + total_ns

    def reset(self, initial_ns: int = 1_000_000_000) -> None:
        self._start_time_ns = initial_ns
        self._current_time_ns = initial_ns
        self._tick_index = 0
        self._frame_id = 0
        self._sequence_id = 0
        self._rate_scale = 1.0
        self._is_paused = False
