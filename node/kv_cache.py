"""
KV Cache Management — quantization, CPU offloading, and prefix caching.

Provides three components:
  1. CacheEntry: wraps DynamicCache with int8 quantization and CPU offloading
  2. KVCacheManager: orchestrates cache lifecycle (create, quantize, offload, clone)

DynamicCache API (transformers >= 4.45):
  - cache.layers[i].keys   — key tensor [batch, heads, seq_len, head_dim]
  - cache.layers[i].values — value tensor [batch, heads, seq_len, head_dim]
  - cache.update(key, value, layer_idx) — append new KV to a layer
  - cache.get_seq_length(layer_idx) — number of cached tokens for a layer

Usage:
    from node.kv_cache import KVCacheManager

    manager = KVCacheManager(quantize="int8", offload_enabled=True)
    cache = manager.get_or_create("session-abc")
    # ... use cache in forward pass ...
    manager.post_forward("session-abc")
    manager.clone_prefix("session-abc", "session-def", prefix_length=500, layer_start=0)
"""

import time
import threading

import torch
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Int8 quantization helpers
# ---------------------------------------------------------------------------

def _quantize_tensor(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to int8 with per-tensor absmax scaling.

    Returns (quantized_int8, scale) where original ~= quantized * scale.
    """
    if t.numel() == 0:
        return t.to(torch.int8), torch.tensor(1.0, device=t.device)
    absmax = t.abs().max().clamp(min=1e-8)
    scale = absmax / 127.0
    quantized = (t / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def _dequantize_tensor(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an int8 tensor back to float32."""
    return quantized.float() * scale


# ---------------------------------------------------------------------------
# Cache entry wrapper
# ---------------------------------------------------------------------------

class CacheEntry:
    """Wraps a DynamicCache with metadata for quantization and offloading."""

    def __init__(self):
        self.cache = DynamicCache()
        self.last_used: float = time.time()
        self.is_quantized: bool = False
        self.is_offloaded: bool = False
        # Quantized storage: list of (key_q, key_scale, value_q, value_scale) per layer
        self._quantized_kv: list[tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]] = []

    def touch(self):
        """Update last-used timestamp."""
        self.last_used = time.time()

    # --- Quantization ---

    def quantize(self):
        """Quantize the KV cache to int8 (saves ~2x memory).

        Skips layers that have None keys/values (placeholder layers created
        by DynamicCache for shards that don't own those layers).
        """
        if self.is_quantized:
            return
        self._quantized_kv = []
        for layer in self.cache.layers:
            if layer.keys is None or layer.values is None:
                # Placeholder layer (not owned by this shard) — skip
                self._quantized_kv.append(None)
                continue
            kq, ks = _quantize_tensor(layer.keys)
            vq, vs = _quantize_tensor(layer.values)
            self._quantized_kv.append((kq, ks, vq, vs))
            # Replace with empty tensors to free memory
            layer.keys = torch.empty(0)
            layer.values = torch.empty(0)
        self.is_quantized = True

    def dequantize(self):
        """Restore the KV cache from int8 back to float32."""
        if not self.is_quantized:
            return
        for i, entry in enumerate(self._quantized_kv):
            if entry is None:
                # Placeholder layer — nothing to dequantize
                continue
            kq, ks, vq, vs = entry
            self.cache.layers[i].keys = _dequantize_tensor(kq, ks)
            self.cache.layers[i].values = _dequantize_tensor(vq, vs)
        self._quantized_kv = []
        self.is_quantized = False

    # --- CPU offloading ---

    def offload_to_cpu(self):
        """Move all cache tensors to CPU RAM."""
        if self.is_offloaded:
            return
        if self.is_quantized:
            self._quantized_kv = [
                ((e[0].cpu(), e[1].cpu(), e[2].cpu(), e[3].cpu()) if e is not None else None)
                for e in self._quantized_kv
            ]
        else:
            for layer in self.cache.layers:
                if layer.keys is not None:
                    layer.keys = layer.keys.cpu()
                if layer.values is not None:
                    layer.values = layer.values.cpu()
        self.is_offloaded = True

    def restore_to_device(self, device: torch.device = None):
        """Move cache tensors back from CPU to the target device."""
        if not self.is_offloaded:
            return
        target = device or torch.device("cpu")
        if self.is_quantized:
            self._quantized_kv = [
                ((e[0].to(target), e[1].to(target), e[2].to(target), e[3].to(target))
                 if e is not None else None)
                for e in self._quantized_kv
            ]
        else:
            for layer in self.cache.layers:
                if layer.keys is not None:
                    layer.keys = layer.keys.to(target)
                if layer.values is not None:
                    layer.values = layer.values.to(target)
        self.is_offloaded = False

    # --- Memory estimation ---

    def estimated_memory_bytes(self) -> int:
        """Estimate the memory used by this cache entry."""
        total = 0
        if self.is_quantized:
            for entry in self._quantized_kv:
                if entry is None:
                    continue
                kq, ks, vq, vs = entry
                total += kq.nelement() * kq.element_size()
                total += vq.nelement() * vq.element_size()
                total += 8  # scales are small
        else:
            for layer in self.cache.layers:
                if layer.keys is not None:
                    total += layer.keys.nelement() * layer.keys.element_size()
                if layer.values is not None:
                    total += layer.values.nelement() * layer.values.element_size()
        return total


# ---------------------------------------------------------------------------
# KV Cache Manager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Orchestrates KV cache lifecycle.

    Args:
        quantize: "none" or "int8"
        offload_enabled: Whether to offload inactive caches to CPU
        offload_after_seconds: Inactivity threshold before offloading
        max_memory_gb: Max memory for all caches (0 = unlimited)
        device: Target device for cache tensors
    """

    def __init__(self, quantize: str = "none", offload_enabled: bool = False,
                 offload_after_seconds: float = 30.0, max_memory_gb: float = 0.0,
                 device: torch.device = None):
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        self.quantize_mode = quantize
        self.offload_enabled = offload_enabled
        self.offload_after_seconds = offload_after_seconds
        self.max_memory_bytes = int(max_memory_gb * 1024**3) if max_memory_gb > 0 else 0
        self.device = device or torch.device("cpu")

    def get_or_create(self, session_id: str) -> DynamicCache:
        """Get an existing cache or create a new one.

        Restores from CPU/dequantizes if needed.
        """
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = CacheEntry()
                self._entries[session_id] = entry

            if entry.is_offloaded:
                entry.restore_to_device(self.device)
            if entry.is_quantized:
                entry.dequantize()

            entry.touch()
            return entry.cache

    def post_forward(self, session_id: str):
        """Called after a forward pass. Quantizes if enabled."""
        with self._lock:
            entry = self._entries.get(session_id)
            if entry and self.quantize_mode == "int8":
                entry.quantize()

    def clone_prefix(self, source_session_id: str, target_session_id: str,
                     prefix_length: int, layer_start: int) -> bool:
        """Clone the first prefix_length tokens of KV cache from source to target.

        Returns True if the clone succeeded.
        """
        with self._lock:
            source = self._entries.get(source_session_id)
            if source is None:
                return False

            # Ensure source is readable
            if source.is_offloaded:
                source.restore_to_device(self.device)
            if source.is_quantized:
                source.dequantize()

            # Check source has enough tokens
            if not source.cache.layers:
                return False
            source_len = source.cache.get_seq_length(layer_start)
            if source_len < prefix_length:
                return False

            # Create target with cloned prefix
            target = CacheEntry()
            for layer in source.cache.layers:
                if layer.keys is None or layer.values is None:
                    # Placeholder layer — create a placeholder in the target too
                    layer_idx = len(target.cache.layers)
                    target.cache.update(
                        torch.zeros(0), torch.zeros(0), layer_idx)
                    # Set to None to match source
                    target.cache.layers[layer_idx].keys = None
                    target.cache.layers[layer_idx].values = None
                    continue
                k = layer.keys[:, :, :prefix_length, :].clone()
                v = layer.values[:, :, :prefix_length, :].clone()
                layer_idx = len(target.cache.layers)
                target.cache.update(k, v, layer_idx)

            # Re-quantize source if needed
            if self.quantize_mode == "int8":
                source.quantize()

            target.touch()
            self._entries[target_session_id] = target
            return True

    def clear_session(self, session_id: str):
        """Remove a session's cache."""
        with self._lock:
            self._entries.pop(session_id, None)

    def cleanup_stale_sessions(self, timeout_seconds: float) -> int:
        """Remove sessions inactive for longer than timeout."""
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, entry in self._entries.items()
                if now - entry.last_used > timeout_seconds
            ]
            for sid in stale:
                del self._entries[sid]
        return len(stale)

    def maybe_offload(self) -> int:
        """Offload inactive sessions to CPU."""
        if not self.offload_enabled:
            return 0
        now = time.time()
        offloaded = 0
        with self._lock:
            for sid, entry in self._entries.items():
                if (not entry.is_offloaded
                        and now - entry.last_used > self.offload_after_seconds):
                    entry.offload_to_cpu()
                    offloaded += 1
        return offloaded

    def maybe_evict_for_memory(self) -> int:
        """Offload LRU sessions if total memory exceeds limit."""
        if self.max_memory_bytes <= 0:
            return 0
        with self._lock:
            total = sum(e.estimated_memory_bytes() for e in self._entries.values())
            if total <= self.max_memory_bytes:
                return 0
            sorted_entries = sorted(
                self._entries.items(), key=lambda x: x[1].last_used)
            evicted = 0
            for sid, entry in sorted_entries:
                if total <= self.max_memory_bytes:
                    break
                if not entry.is_offloaded:
                    mem = entry.estimated_memory_bytes()
                    entry.offload_to_cpu()
                    total -= mem
                    evicted += 1
            return evicted

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total_mem = sum(e.estimated_memory_bytes() for e in self._entries.values())
            return {
                "sessions": len(self._entries),
                "quantized": sum(1 for e in self._entries.values() if e.is_quantized),
                "offloaded": sum(1 for e in self._entries.values() if e.is_offloaded),
                "total_memory_mb": round(total_mem / (1024 * 1024), 2),
            }
