"""
Hardware capacity probing and verification for UNFED auto-assignment.

Detects CPU, RAM, and GPU resources so the registry can assign an
appropriate role/shard. Also implements the challenge-response protocol
for capacity verification (prove you actually have the reported resources).
"""

import hashlib
import os
import time
from typing import Optional


def probe_capacity() -> dict:
    """Auto-detect hardware capacity.

    Returns a dict matching the NodeCapacity proto fields:
        total_ram_gb, available_ram_gb, cpu_cores,
        has_gpu, gpu_vram_gb, available_gpu_vram_gb
    """
    return {
        "total_ram_gb": _get_total_ram_gb(),
        "available_ram_gb": _get_available_ram_gb(),
        "cpu_cores": os.cpu_count() or 1,
        "has_gpu": _has_gpu(),
        "gpu_vram_gb": _get_gpu_vram_gb(),
        "available_gpu_vram_gb": _get_available_gpu_vram_gb(),
    }


def list_available_shards(shards_dir: str) -> list[str]:
    """List shard weight files (*.pt, *.safetensors) in the given directory."""
    if not os.path.isdir(shards_dir):
        return []
    return sorted(
        f for f in os.listdir(shards_dir)
        if f.endswith(".pt") or f.endswith(".safetensors")
    )


def respond_to_challenge(
    challenge_bytes: int,
    device: str = "cpu",
) -> tuple[str, float]:
    """Allocate challenge_bytes on device, fill deterministically, and hash.

    Returns (sha256_hex, allocation_time_ms).
    Proves the node actually has the reported memory available.
    """
    start = time.perf_counter()

    if device == "cuda":
        try:
            import torch
            buf = torch.zeros(challenge_bytes // 4, dtype=torch.float32, device="cuda")
            # Deterministic fill: index-based pattern
            buf[:] = torch.arange(buf.numel(), dtype=torch.float32, device="cuda") % 256
            torch.cuda.synchronize()
            data = buf.cpu().numpy().tobytes()
            del buf
        except Exception:
            return "", -1.0
    else:
        buf = bytearray(challenge_bytes)
        for i in range(0, len(buf), 4096):
            end = min(i + 4096, len(buf))
            for j in range(i, end):
                buf[j] = j % 256
        data = bytes(buf)
        del buf

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    proof_hash = hashlib.sha256(data).hexdigest()
    return proof_hash, elapsed_ms


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    # Try /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        pass

    # Fallback: os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024 ** 3)
    except (ValueError, OSError):
        pass

    return 0.0


def _get_available_ram_gb() -> float:
    """Get available (free + cached) system RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        pass

    # Fallback: os.sysconf (free pages only, no cached)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024 ** 3)
    except (ValueError, OSError):
        pass

    return 0.0


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_gpu_vram_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        return props.total_mem / (1024 ** 3)
    except (ImportError, RuntimeError):
        return 0.0


def _get_available_gpu_vram_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    except (ImportError, RuntimeError):
        return 0.0
