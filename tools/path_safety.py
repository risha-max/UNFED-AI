"""Path safety helpers for manifest/tool file access."""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a path escapes allowed boundaries."""


def safe_join_under(base_dir: str, relative_path: str) -> Path:
    """Resolve a manifest-relative path while confining it under base_dir."""
    base = Path(base_dir).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise UnsafePathError(f"absolute path not allowed: {relative_path}")

    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise UnsafePathError(
            f"path escapes base directory: {relative_path}"
        ) from exc
    return resolved
