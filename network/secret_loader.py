"""
Helpers for loading sensitive values from env or files.
"""

from __future__ import annotations

import os


def load_secret_from_env(
    env_var: str,
    *,
    file_env_var: str | None = None,
) -> str:
    direct = (os.getenv(env_var, "") or "").strip()
    if direct:
        return direct
    if not file_env_var:
        return ""
    path = (os.getenv(file_env_var, "") or "").strip()
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""
