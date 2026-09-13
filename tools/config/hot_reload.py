"""Hot-reload validation, safe field allowlist enforcement, and tuning parameter clamping."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Blueprint-defined safe numeric tuning parameters permitted to hot-reload without quiescent restart
ALLOWLISTED_HOT_RELOAD_FIELDS: frozenset[str] = frozenset(
    [
        "perception.confidence_floor",
        "perception.nms_iou_threshold",
        "tracking.max_missed_frames",
        "tracking.immediate_confidence",
        "tracking.require_second_observation_below",
        "tracking.engage_if_uncertainty_within_radius",
        "tracking.process_noise_scale",
        "tracking.measurement_noise_scale",
        "tracking.gating_threshold_chi2",
        "policy.switch_hysteresis",
        "policy.horizon_targets",
        "policy.target_lead_time_ms",
        "runtime.internal_deadline_ms",
        "runtime.target_p99_ms",
        "safety.fail_closed",
        "safety.max_delta_counts_per_dispatch",
        "safety.max_active_engagement_seconds",
    ]
)


@dataclass
class HotReloadResult:
    """Diagnostic outcome of checking configuration hot-reload compatibility."""

    is_allowed: bool = True
    requires_restart: bool = False
    modified_allowed_fields: list[str] = field(default_factory=list)
    conflicting_structural_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary using dot-separated path keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def validate_hot_reload(current_cfg: dict[str, Any], new_cfg: dict[str, Any]) -> HotReloadResult:
    """Compare active configuration against proposed updated configuration.

    Returns HotReloadResult indicating whether the change is safely hot-reloadable
    or requires a quiescent application restart.
    """
    flat_current = flatten_dict(current_cfg)
    flat_new = flatten_dict(new_cfg)

    all_keys = sorted(set(flat_current.keys()) | set(flat_new.keys()))
    modified_allowed: list[str] = []
    conflicting_structural: list[str] = []

    for key in all_keys:
        curr_val = flat_current.get(key)
        new_val = flat_new.get(key)

        if curr_val != new_val:
            if key in ALLOWLISTED_HOT_RELOAD_FIELDS:
                modified_allowed.append(key)
            else:
                conflicting_structural.append(key)

    requires_restart = len(conflicting_structural) > 0
    is_allowed = not requires_restart

    return HotReloadResult(
        is_allowed=is_allowed,
        requires_restart=requires_restart,
        modified_allowed_fields=modified_allowed,
        conflicting_structural_fields=conflicting_structural,
    )


def clamp_tuning_parameters(config_dict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clamp continuous tuning parameters within nominal operating bounds.

    Returns (clamped_config_copy, warning_messages).
    """
    cfg = copy.deepcopy(config_dict)
    warnings: list[str] = []

    def _clamp_float(
        domain: str, field_name: str, min_val: float, max_val: float
    ) -> None:
        if domain in cfg and isinstance(cfg[domain], dict) and field_name in cfg[domain]:
            val = cfg[domain][field_name]
            if isinstance(val, (int, float)):
                if val < min_val:
                    warnings.append(
                        f"Clamped {domain}.{field_name} from {val} to nominal minimum {min_val}"
                    )
                    cfg[domain][field_name] = type(val)(min_val)
                elif val > max_val:
                    warnings.append(
                        f"Clamped {domain}.{field_name} from {val} to nominal maximum {max_val}"
                    )
                    cfg[domain][field_name] = type(val)(max_val)

    # Perception tuning bounds
    _clamp_float("perception", "confidence_floor", 0.0, 1.0)
    _clamp_float("perception", "nms_iou_threshold", 0.0, 1.0)

    # Tracking tuning bounds
    _clamp_float("tracking", "immediate_confidence", 0.0, 1.0)
    _clamp_float("tracking", "require_second_observation_below", 0.0, 1.0)

    # Policy tuning bounds
    _clamp_float("policy", "switch_hysteresis", 0.0, 10.0)

    # Runtime tuning bounds
    _clamp_float("runtime", "warmup_iterations", 0, 10000)

    # Trajectory tuning bounds
    _clamp_float("trajectory", "small_error_threshold_px", 0.0, 500.0)
    _clamp_float("trajectory", "max_velocity_counts_per_s", 100.0, 1000000.0)
    _clamp_float("trajectory", "max_acceleration_counts_per_s2", 1000.0, 100000000.0)
    _clamp_float("trajectory", "max_jerk_counts_per_s3", 10000.0, 1000000000.0)

    # Actuator scheduler bounds
    _clamp_float("actuator", "scheduler_hz", 100, 8000)

    return cfg, warnings
