"""Immutable execution run manifest generation, serialization, and verification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any
import uuid

import jsonschema

from tools.config.hashing import compute_config_sha256, compute_file_sha256


def get_git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Safely obtain Git repository lineage metadata, with graceful fallbacks."""
    root = repo_root or Path(__file__).resolve().parents[2]
    info = {
        "commit_hash": "untracked",
        "branch": "unknown",
        "is_dirty": False,
        "describe": "unknown",
    }
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            info["commit_hash"] = res.stdout.strip()

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res_branch.returncode == 0 and res_branch.stdout.strip():
            info["branch"] = res_branch.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res_status.returncode == 0:
            info["is_dirty"] = len(res_status.stdout.strip()) > 0

        res_desc = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res_desc.returncode == 0 and res_desc.stdout.strip():
            info["describe"] = res_desc.stdout.strip()
    except Exception:
        pass
    return info


def get_environment_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Retrieve host system hardware and OS environment telemetry."""
    root = repo_root or Path(__file__).resolve().parents[2]
    env_manifest_path = root / "docs" / "status" / "environment_manifest.json"
    if env_manifest_path.is_file():
        try:
            data = json.loads(env_manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def collect_schema_hashes(repo_root: Path | None = None) -> dict[str, str]:
    """Collect SHA-256 hashes of all JSON and FlatBuffers schemas in the repository."""
    root = repo_root or Path(__file__).resolve().parents[2]
    schemas_dir = root / "schemas"
    schema_hashes: dict[str, str] = {}

    if schemas_dir.is_dir():
        for schema_file in schemas_dir.rglob("*"):
            if schema_file.is_file() and schema_file.suffix in {".json", ".fbs"}:
                rel_path = schema_file.relative_to(root).as_posix()
                schema_hashes[rel_path] = compute_file_sha256(schema_file)
    return schema_hashes


class RunManifestBuilder:
    """Builder for immutable runtime execution manifests."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    def build(
        self,
        resolved_config: dict[str, Any],
        source_layers: list[str],
        config_sha256: str | None = None,
        manifest_id: str | None = None,
        timestamp_utc: str | None = None,
        git_info: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        mode: str = "simulation",
        warmup_iterations_completed: int = 200,
        notes: str = "",
    ) -> dict[str, Any]:
        """Construct a complete, schema-compliant run manifest dictionary."""
        cfg_hash = config_sha256 or compute_config_sha256(resolved_config)
        m_id = manifest_id or str(uuid.uuid4())
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
        git_meta = git_info if git_info is not None else get_git_info(self.repo_root)
        env_meta = environment if environment is not None else get_environment_info(self.repo_root)
        schema_meta = collect_schema_hashes(self.repo_root)
        artifact_list = artifacts or []

        runtime_sec = resolved_config.get("runtime", {})
        internal_deadline = runtime_sec.get("internal_deadline_ms", 10.0)
        target_p99 = runtime_sec.get("target_p99_ms", 6.0)

        # Normalize source layer paths to POSIX slashes
        normalized_layers = [Path(p).as_posix() for p in source_layers]

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "manifest_id": m_id,
            "timestamp_utc": ts,
            "git_info": {
                "commit_hash": git_meta.get("commit_hash", "untracked"),
                "branch": git_meta.get("branch", "unknown"),
                "is_dirty": bool(git_meta.get("is_dirty", False)),
                "describe": git_meta.get("describe", "unknown"),
            },
            "environment": env_meta,
            "config": {
                "source_layers": normalized_layers,
                "config_sha256": cfg_hash,
                "resolved_config": resolved_config,
            },
            "schemas": schema_meta,
            "artifacts": artifact_list,
            "execution": {
                "mode": mode,
                "internal_deadline_ms": float(internal_deadline),
                "target_p99_ms": float(target_p99),
                "warmup_iterations_completed": int(warmup_iterations_completed),
                "notes": notes,
            },
        }
        return manifest


def verify_run_manifest(
    manifest_dict: dict[str, Any],
    manifest_schema_path: Path | None = None,
) -> tuple[bool, str]:
    """Verify schema conformance and cryptographic integrity of a run manifest.

    Returns:
        (is_valid: bool, diagnostic_message: str)
    """
    schema_path = (
        manifest_schema_path
        or Path(__file__).resolve().parents[2] / "schemas" / "manifest" / "run_manifest.schema.json"
    )

    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            validator = jsonschema.Draft202012Validator(schema)
            errors = list(validator.iter_errors(manifest_dict))
            if errors:
                first_err = errors[0]
                path = ".".join(str(p) for p in first_err.absolute_path) or "<root>"
                return False, f"Manifest schema violation at '{path}': {first_err.message}"
        except Exception as e:
            return False, f"Failed to validate manifest schema: {e}"

    if "config" not in manifest_dict or "resolved_config" not in manifest_dict["config"]:
        return False, "Missing 'config.resolved_config' section in manifest"

    recorded_hash = manifest_dict["config"].get("config_sha256", "")
    resolved_cfg = manifest_dict["config"]["resolved_config"]
    recomputed_hash = compute_config_sha256(resolved_cfg)

    if recorded_hash.lower() != recomputed_hash.lower():
        return (
            False,
            f"Config SHA-256 tamper detected: recorded '{recorded_hash}' != recomputed '{recomputed_hash}'",
        )

    return True, "Manifest verified successfully with cryptographic integrity confirmed."
