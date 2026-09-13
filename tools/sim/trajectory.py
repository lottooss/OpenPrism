"""tools/sim/trajectory.py
Kinematic trajectory evaluation and boundary reflection algorithms.

This module mirrors ``include/aim/sim/trajectory.hpp`` operation for operation. The C++ side
evaluates every kinematic expression in ``float`` (IEEE-754 binary32) and rounds after each
operation; Python arithmetic is binary64. Each expression below therefore rounds through
``f32`` at exactly the points the C++ compiler does, so both implementations produce the same
bit pattern for the same scenario. ``tests/golden/cross_language_sim_parity.py`` enforces it.
"""

from __future__ import annotations

import math
from typing import Tuple

from tools.bus.schema_bindings import Vec2f
from tools.sim.prng import f32
from tools.sim.types import BoundaryBehavior, TargetSpec, TrajectoryState, TrajectoryType

_TWO_PI_F32 = f32(6.28318530717958647692)
_PI_F32 = f32(3.14159265358979323846)


def _add(lhs: float, rhs: float) -> float:
    """Single binary32 addition."""
    return f32(lhs + rhs)


def _sub(lhs: float, rhs: float) -> float:
    """Single binary32 subtraction."""
    return f32(lhs - rhs)


def _mul(lhs: float, rhs: float) -> float:
    """Single binary32 multiplication."""
    return f32(lhs * rhs)


def sin_f32(radians: float) -> float:
    """binary32 sine, matching aim::sim::sin_f32 (binary64 evaluation, narrowed once)."""
    return f32(math.sin(radians))


def cos_f32(radians: float) -> float:
    """binary32 cosine, matching aim::sim::cos_f32 (binary64 evaluation, narrowed once)."""
    return f32(math.cos(radians))


def apply_1d_boundary(
    raw_pos: float,
    raw_vel: float,
    min_val: float,
    max_val: float,
    behavior: BoundaryBehavior,
) -> Tuple[float, float, bool]:
    """Applies boundary interaction for a 1D coordinate within [min_val, max_val]."""
    raw_pos, raw_vel = f32(raw_pos), f32(raw_vel)
    min_val, max_val = f32(min_val), f32(max_val)

    span = _sub(max_val, min_val)
    if span <= 0.0:
        return (min_val, 0.0, False)

    despawned = False
    out_pos = raw_pos
    out_vel = raw_vel

    if behavior == BoundaryBehavior.BOUNCE:
        u = _sub(raw_pos, min_val)
        period = _mul(2.0, span)
        m = f32(math.fmod(u, period))
        if m < 0.0:
            m = _add(m, period)
        if m <= span:
            out_pos = _add(min_val, m)
            out_vel = raw_vel
        else:
            out_pos = _sub(_add(min_val, _mul(2.0, span)), m)
            out_vel = -raw_vel

    elif behavior == BoundaryBehavior.WRAP:
        u = _sub(raw_pos, min_val)
        m = f32(math.fmod(u, span))
        if m < 0.0:
            m = _add(m, span)
        out_pos = _add(min_val, m)
        out_vel = raw_vel

    elif behavior == BoundaryBehavior.CLAMP:
        if raw_pos < min_val:
            out_pos = min_val
            out_vel = 0.0
        elif raw_pos > max_val:
            out_pos = max_val
            out_vel = 0.0

    elif behavior == BoundaryBehavior.DESPAWN:
        despawned = raw_pos < min_val or raw_pos > max_val

    return (out_pos, out_vel, despawned)


def evaluate_trajectory(
    spec: TargetSpec,
    elapsed_sec: float,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> TrajectoryState:
    """Evaluates target state for a given elapsed simulation time in seconds."""
    radius = f32(spec.radius_px)
    state = TrajectoryState(
        radius_px=radius,
        is_active=True,
        is_despawned=False,
    )

    # C++ narrows the double elapsed time to float once, then works entirely in float.
    t = f32(elapsed_sec)
    init_x, init_y = f32(spec.initial_pos_px.x), f32(spec.initial_pos_px.y)
    vel_x, vel_y = f32(spec.velocity_px_s.x), f32(spec.velocity_px_s.y)
    acc_x, acc_y = f32(spec.accel_px_s2.x), f32(spec.accel_px_s2.y)
    min_x = radius
    max_x = _sub(f32(screen_width), radius)
    min_y = radius
    max_y = _sub(f32(screen_height), radius)

    if spec.trajectory_type == TrajectoryType.STATIONARY:
        state.pos_px = Vec2f(init_x, init_y)
        state.vel_px_per_s = Vec2f(0.0, 0.0)
        state.accel_px_s2 = Vec2f(0.0, 0.0)

    elif spec.trajectory_type in (TrajectoryType.LINEAR_CV, TrajectoryType.BOUNCING_BOX):
        raw_x = _add(init_x, _mul(vel_x, t))
        raw_y = _add(init_y, _mul(vel_y, t))

        px_x, vx, desp_x = apply_1d_boundary(raw_x, vel_x, min_x, max_x, spec.boundary)
        px_y, vy, desp_y = apply_1d_boundary(raw_y, vel_y, min_y, max_y, spec.boundary)

        state.pos_px = Vec2f(px_x, px_y)
        state.vel_px_per_s = Vec2f(vx, vy)
        state.accel_px_s2 = Vec2f(0.0, 0.0)
        if desp_x or desp_y:
            state.is_despawned = True
            state.is_active = False

    elif spec.trajectory_type == TrajectoryType.LINEAR_CA:
        raw_x = _add(_add(init_x, _mul(vel_x, t)), _mul(_mul(_mul(0.5, acc_x), t), t))
        raw_y = _add(_add(init_y, _mul(vel_y, t)), _mul(_mul(_mul(0.5, acc_y), t), t))
        raw_vx = _add(vel_x, _mul(acc_x, t))
        raw_vy = _add(vel_y, _mul(acc_y, t))

        px_x, vx, desp_x = apply_1d_boundary(raw_x, raw_vx, min_x, max_x, spec.boundary)
        px_y, vy, desp_y = apply_1d_boundary(raw_y, raw_vy, min_y, max_y, spec.boundary)

        state.pos_px = Vec2f(px_x, px_y)
        state.vel_px_per_s = Vec2f(vx, vy)
        state.accel_px_s2 = Vec2f(acc_x, acc_y)
        if desp_x or desp_y:
            state.is_despawned = True
            state.is_active = False

    elif spec.trajectory_type == TrajectoryType.SINUSOIDAL_STRAFE:
        amp_x, amp_y = f32(spec.amplitude_x_px), f32(spec.amplitude_y_px)
        freq = f32(spec.frequency_hz)
        phase = f32(spec.phase_rad)
        omega_x = _mul(_TWO_PI_F32, freq)
        omega_y = _mul(_mul(_TWO_PI_F32, freq), 0.5) if amp_y > 0.0 else 0.0

        arg_x = _add(_mul(omega_x, t), phase)
        arg_y = _add(_mul(omega_y, t), phase)

        state.pos_px = Vec2f(
            _add(init_x, _mul(amp_x, sin_f32(arg_x))),
            _add(init_y, _mul(amp_y, cos_f32(arg_y))),
        )
        state.vel_px_per_s = Vec2f(
            _mul(_mul(amp_x, omega_x), cos_f32(arg_x)),
            _mul(_mul(-amp_y, omega_y), sin_f32(arg_y)),
        )
        state.accel_px_s2 = Vec2f(
            _mul(_mul(_mul(-amp_x, omega_x), omega_x), sin_f32(arg_x)),
            _mul(_mul(_mul(-amp_y, omega_y), omega_y), cos_f32(arg_y)),
        )

    elif spec.trajectory_type == TrajectoryType.CIRCULAR_ORBIT:
        omega = _mul(_TWO_PI_F32, f32(spec.frequency_hz))
        radius_orbit = f32(spec.amplitude_x_px)
        angle = _add(_mul(omega, t), f32(spec.phase_rad))

        state.pos_px = Vec2f(
            _add(init_x, _mul(radius_orbit, cos_f32(angle))),
            _add(init_y, _mul(radius_orbit, sin_f32(angle))),
        )
        state.vel_px_per_s = Vec2f(
            _mul(_mul(-radius_orbit, omega), sin_f32(angle)),
            _mul(_mul(radius_orbit, omega), cos_f32(angle)),
        )
        state.accel_px_s2 = Vec2f(
            _mul(_mul(_mul(-radius_orbit, omega), omega), cos_f32(angle)),
            _mul(_mul(_mul(-radius_orbit, omega), omega), sin_f32(angle)),
        )

    elif spec.trajectory_type == TrajectoryType.SUDDEN_CUT:
        # Step selection stays in double precision, exactly as the C++ path does.
        cut_interval_s = float(spec.cut_interval_ns) / 1_000_000_000.0
        if cut_interval_s <= 0.0:
            state.pos_px = Vec2f(_add(init_x, _mul(vel_x, t)), _add(init_y, _mul(vel_y, t)))
            state.vel_px_per_s = Vec2f(vel_x, vel_y)
            state.accel_px_s2 = Vec2f(0.0, 0.0)
            return state

        step_index = int(elapsed_sec / cut_interval_s)
        remainder_s = elapsed_sec - float(step_index) * cut_interval_s
        dt_rem = f32(remainder_s)
        cut_dt = f32(cut_interval_s)

        speed = f32(math.sqrt(_add(_mul(vel_x, vel_x), _mul(vel_y, vel_y))))
        theta0 = f32(math.atan2(vel_y, vel_x)) if speed > 1e-6 else 0.0
        cut_angle = f32(spec.cut_angle_rad)

        if abs(_sub(cut_angle, _PI_F32)) < f32(1e-4):
            if (step_index % 2) == 0:
                state.pos_px = Vec2f(_add(init_x, _mul(vel_x, dt_rem)), _add(init_y, _mul(vel_y, dt_rem)))
                state.vel_px_per_s = Vec2f(vel_x, vel_y)
            else:
                back = _sub(cut_dt, dt_rem)
                state.pos_px = Vec2f(_add(init_x, _mul(vel_x, back)), _add(init_y, _mul(vel_y, back)))
                state.vel_px_per_s = Vec2f(-vel_x, -vel_y)
        else:
            cur_x, cur_y = init_x, init_y
            for j in range(step_index):
                angle_j = _add(theta0, _mul(f32(j), cut_angle))
                cur_x = _add(cur_x, _mul(_mul(speed, cos_f32(angle_j)), cut_dt))
                cur_y = _add(cur_y, _mul(_mul(speed, sin_f32(angle_j)), cut_dt))

            angle_cur = _add(theta0, _mul(f32(step_index), cut_angle))
            cur_vx = _mul(speed, cos_f32(angle_cur))
            cur_vy = _mul(speed, sin_f32(angle_cur))
            state.pos_px = Vec2f(_add(cur_x, _mul(cur_vx, dt_rem)), _add(cur_y, _mul(cur_vy, dt_rem)))
            state.vel_px_per_s = Vec2f(cur_vx, cur_vy)

        state.accel_px_s2 = Vec2f(0.0, 0.0)

    else:
        state.pos_px = Vec2f(init_x, init_y)
        state.vel_px_per_s = Vec2f(vel_x, vel_y)
        state.accel_px_s2 = Vec2f(acc_x, acc_y)

    return state
