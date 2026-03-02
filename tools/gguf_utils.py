"""
GGUF helpers with defensive parsing guardrails.

This module centralizes GGUF parsing so converter/inspector/splitter reuse the
same limits and validation behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# Defensive caps to avoid unbounded allocations on malformed files.
MAX_GGUF_FILE_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB
MAX_GGUF_TENSORS = 200_000
MAX_GGUF_TOTAL_TENSOR_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB
MAX_GGUF_SINGLE_TENSOR_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB

_SUPPORTED_TORCH_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float32,  # downcast float64 for portability
}


class GGUFError(ValueError):
    """Raised when a GGUF file fails validation or contains unsupported data."""


def _require_gguf_dependency():
    try:
        from gguf import GGUFReader  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise GGUFError(
            "GGUF support requires the 'gguf' package. Install with: pip install gguf"
        ) from exc


def _validate_input_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise GGUFError(f"GGUF file not found: {path}")
    if p.suffix.lower() != ".gguf":
        raise GGUFError(f"Expected a .gguf file, got: {p.suffix or '<no extension>'}")
    size = p.stat().st_size
    if size <= 0:
        raise GGUFError("GGUF file is empty")
    if size > MAX_GGUF_FILE_BYTES:
        raise GGUFError(
            f"GGUF file too large ({size} bytes). Limit is {MAX_GGUF_FILE_BYTES} bytes."
        )
    return p


def _field_value(reader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """Read lightweight GGUF metadata and tensor summary."""
    _require_gguf_dependency()
    gguf_path = _validate_input_path(path)

    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    tensor_count = len(reader.tensors)
    if tensor_count > MAX_GGUF_TENSORS:
        raise GGUFError(
            f"GGUF tensor count too large ({tensor_count}). Limit is {MAX_GGUF_TENSORS}."
        )

    total_bytes = 0
    dtypes: dict[str, int] = {}
    for t in reader.tensors:
        n_bytes = int(getattr(t, "n_bytes", 0))
        if n_bytes <= 0:
            raise GGUFError(f"Tensor '{t.name}' has invalid byte size: {n_bytes}")
        if n_bytes > MAX_GGUF_SINGLE_TENSOR_BYTES:
            raise GGUFError(
                f"Tensor '{t.name}' exceeds per-tensor limit ({n_bytes} bytes)."
            )
        total_bytes += n_bytes
        dtype_name = str(np.dtype(t.data.dtype))
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1

    if total_bytes > MAX_GGUF_TOTAL_TENSOR_BYTES:
        raise GGUFError(
            f"GGUF tensor payload too large ({total_bytes} bytes). "
            f"Limit is {MAX_GGUF_TOTAL_TENSOR_BYTES} bytes."
        )

    return {
        "path": str(gguf_path),
        "architecture": _field_value(reader, "general.architecture", "unknown"),
        "name": _field_value(reader, "general.name", ""),
        "file_type": _field_value(reader, "general.file_type", None),
        "tensor_count": tensor_count,
        "total_tensor_bytes": total_bytes,
        "tensor_dtypes": dtypes,
        "supported_tensor_dtypes": sorted(str(k) for k in _SUPPORTED_TORCH_DTYPES.keys()),
    }


def gguf_weight_keys(path: str) -> dict[str, tuple[torch.Size, str]]:
    """Return GGUF tensor metadata as key -> (shape, dtype)."""
    _require_gguf_dependency()
    gguf_path = _validate_input_path(path)

    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    if len(reader.tensors) > MAX_GGUF_TENSORS:
        raise GGUFError("GGUF has too many tensors")

    result: dict[str, tuple[torch.Size, str]] = {}
    for t in reader.tensors:
        name = str(t.name)
        if not name:
            raise GGUFError("Encountered unnamed tensor in GGUF")
        if "/" in name or "\\" in name:
            raise GGUFError(f"Invalid tensor name '{name}' (contains path separator)")
        shape = tuple(int(x) for x in getattr(t.data, "shape", ()))
        result[name] = (torch.Size(shape), str(np.dtype(t.data.dtype)))
    return result


def gguf_to_state_dict(path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load supported GGUF tensors into a CPU torch state_dict.

    Only non-quantized float tensors are accepted for conversion.
    """
    _require_gguf_dependency()
    meta = read_gguf_metadata(path)

    from gguf import GGUFReader

    reader = GGUFReader(meta["path"])
    state_dict: dict[str, torch.Tensor] = {}
    seen_names: set[str] = set()

    for t in reader.tensors:
        name = str(t.name)
        if name in seen_names:
            raise GGUFError(f"Duplicate tensor name in GGUF: {name}")
        seen_names.add(name)

        np_dtype = np.dtype(t.data.dtype)
        torch_dtype = _SUPPORTED_TORCH_DTYPES.get(np_dtype)
        if torch_dtype is None:
            raise GGUFError(
                f"Unsupported GGUF tensor dtype for '{name}': {np_dtype}. "
                "Quantized GGUF tensors are not supported by this converter yet."
            )

        np_array = np.array(t.data, copy=True)
        tensor = torch.from_numpy(np_array).to(dtype=torch_dtype).contiguous()
        state_dict[name] = tensor

    return state_dict, meta
