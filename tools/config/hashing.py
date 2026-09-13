"""Canonical JSON serialization and SHA-256 hashing for configuration and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_dumps(obj: Any) -> str:
    """Serialize an object to a canonical JSON string.

    Uses lexicographically sorted keys and compact separators without whitespace
    to guarantee byte-for-byte reproducibility across platforms and languages.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_sha256(config_dict: dict[str, Any]) -> str:
    """Compute the SHA-256 hexadecimal digest of a configuration dictionary."""
    canonical_str = canonical_json_dumps(config_dict)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest().lower()


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file on disk."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_string_sha256(content: str | bytes) -> str:
    """Compute the SHA-256 hexadecimal digest of a string or byte sequence."""
    if isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = content
    return hashlib.sha256(data).hexdigest().lower()
