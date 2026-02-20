"""
UNFED Shard Verifier — validate shard integrity and correctness.

Checks:
  1. Manifest structure — required fields, valid shard references
  2. File existence — all referenced shard files present
  3. SHA256 hashes — file hashes match manifest
  4. Chunk hashes — per-piece P2P hashes match
  5. Weight keys — shard files contain expected weight tensors
  6. Verification vectors — deterministic embedding test (if present)
  7. Deep verify — compare against original model (optional)

Usage:
    python -m tools verify ./shards/manifest.json
    python -m tools verify ./shards/manifest.json --deep --source /path/to/model
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch


def _get_all_shards(manifest: dict) -> list[dict]:
    """Get all shards from either the v3 unified array or legacy v2 arrays."""
    if manifest.get("shards"):
        return manifest["shards"]
    return manifest.get("text_shards", []) + manifest.get("vision_shards", [])


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class VerifyResult:
    """Tracks verification results."""

    def __init__(self):
        self.checks: list[dict] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def warn(self, name: str, detail: str = ""):
        self.checks.append({"name": name, "status": "WARN", "detail": detail})
        self.warnings += 1

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        return (f"{self.passed} passed, {self.failed} failed, "
                f"{self.warnings} warnings")


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_chunk_hashes(path: str, chunk_size: int) -> list[str]:
    """Compute per-chunk SHA256 hashes."""
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


# ---------------------------------------------------------------------------
# Verification steps
# ---------------------------------------------------------------------------

def verify_manifest_structure(manifest: dict, result: VerifyResult):
    """Check that the manifest has all required fields."""
    # Required top-level fields
    has_shards = bool(manifest.get("shards") or manifest.get("text_shards"))
    for field in ("model_id", "model_type"):
        result.check(
            f"manifest.{field}",
            field in manifest,
            f"{'present' if field in manifest else 'MISSING'}",
        )
    result.check(
        "manifest.shards",
        has_shards,
        "present" if has_shards else "MISSING (need 'shards' or 'text_shards')",
    )

    # Shard structure (works with both v3 unified and v2 legacy arrays)
    all_shards = _get_all_shards(manifest)
    for i, shard in enumerate(all_shards):
        label = shard.get("stack", shard.get("file", f"shard_{i}"))
        for field in ("shard_index", "layer_start", "layer_end", "file", "sha256"):
            result.check(
                f"shard[{i}:{label}].{field}",
                field in shard,
                f"{'present' if field in shard else 'MISSING'}",
            )

    # Architecture config (v2+ manifests)
    if manifest.get("format_version", 1) >= 2:
        arch = manifest.get("architecture", {})
        result.check(
            "architecture.text",
            bool(arch.get("text")),
            f"{'present' if arch.get('text') else 'MISSING (v2+ manifest should have this)'}",
        )

    # v3 stacks metadata
    if manifest.get("format_version", 1) >= 3:
        stacks = manifest.get("stacks", {})
        result.check(
            "manifest.stacks",
            bool(stacks),
            f"{len(stacks)} stack(s)" if stacks else "MISSING (v3 manifest should have this)",
        )


def verify_files_exist(manifest: dict, base_dir: str, result: VerifyResult):
    """Check that all referenced shard files exist."""
    all_shards = _get_all_shards(manifest)
    for shard in all_shards:
        fname = shard.get("file", "")
        path = os.path.join(base_dir, fname)
        exists = os.path.isfile(path)
        size_ok = True
        detail = ""
        if exists:
            actual_size = os.path.getsize(path)
            expected_size = shard.get("size_bytes")
            if expected_size and actual_size != expected_size:
                size_ok = False
                detail = (f"size mismatch: expected {expected_size:,} bytes, "
                          f"got {actual_size:,} bytes")
            else:
                detail = f"{actual_size / (1024 * 1024):.1f} MB"
        else:
            detail = f"file not found: {path}"

        result.check(f"file_exists({fname})", exists, detail)
        if exists and not size_ok:
            result.check(f"file_size({fname})", False, detail)


def verify_hashes(manifest: dict, base_dir: str, result: VerifyResult):
    """Verify SHA256 hashes of all shard files."""
    all_shards = _get_all_shards(manifest)
    for shard in all_shards:
        fname = shard.get("file", "")
        path = os.path.join(base_dir, fname)
        expected_hash = shard.get("sha256", "")

        if not os.path.isfile(path):
            result.warn(f"hash({fname})", "skipped — file not found")
            continue

        if not expected_hash:
            result.warn(f"hash({fname})", "no expected hash in manifest")
            continue

        actual_hash = compute_file_hash(path)
        match = actual_hash == expected_hash
        detail = "match" if match else f"MISMATCH: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        result.check(f"sha256({fname})", match, detail)


def verify_chunk_hashes(manifest: dict, base_dir: str, result: VerifyResult):
    """Verify per-chunk hashes for P2P integrity."""
    all_shards = _get_all_shards(manifest)
    for shard in all_shards:
        fname = shard.get("file", "")
        path = os.path.join(base_dir, fname)
        expected_chunks = shard.get("chunk_hashes", [])
        chunk_size = shard.get("chunk_size", 4 * 1024 * 1024)

        if not expected_chunks:
            continue  # No chunk hashes to verify

        if not os.path.isfile(path):
            result.warn(f"chunks({fname})", "skipped — file not found")
            continue

        actual_chunks = compute_chunk_hashes(path, chunk_size)

        if len(actual_chunks) != len(expected_chunks):
            result.check(
                f"chunks({fname})",
                False,
                f"chunk count mismatch: expected {len(expected_chunks)}, "
                f"got {len(actual_chunks)}",
            )
            continue

        mismatches = []
        for i, (expected, actual) in enumerate(zip(expected_chunks, actual_chunks)):
            if expected != actual:
                mismatches.append(i)

        if mismatches:
            result.check(
                f"chunks({fname})",
                False,
                f"{len(mismatches)} chunk(s) corrupted: {mismatches[:5]}...",
            )
        else:
            result.check(
                f"chunks({fname})",
                True,
                f"all {len(actual_chunks)} chunks verified",
            )


def verify_weight_keys(manifest: dict, base_dir: str, result: VerifyResult):
    """Verify shard files contain expected weight tensors."""
    all_shards = _get_all_shards(manifest)

    for shard in all_shards:
        fname = shard.get("file", "")
        path = os.path.join(base_dir, fname)

        if not os.path.isfile(path):
            continue

        try:
            ext = Path(path).suffix.lower()
            if ext == ".safetensors":
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
            else:
                sd = torch.load(path, map_location="cpu", weights_only=True)
                keys = list(sd.keys()) if isinstance(sd, dict) else []

            result.check(
                f"weights({fname})",
                len(keys) > 0,
                f"{len(keys)} tensors",
            )

            # Check for expected components based on shard metadata
            has_embed = shard.get("has_embedding") or shard.get("has_embeddings")
            has_head = shard.get("has_lm_head") or shard.get("has_connector")
            layer_start = shard.get("layer_start", 0)
            layer_end = shard.get("layer_end", 0)

            if has_embed:
                has_embed_key = any("embed" in k for k in keys)
                result.check(
                    f"weights({fname}).embedding",
                    has_embed_key,
                    f"{'found' if has_embed_key else 'MISSING — manifest says has_embedding=true'}",
                )

            if has_head:
                has_head_key = any(
                    "lm_head" in k or "connector" in k or "post_layernorm" in k
                    for k in keys
                )
                result.check(
                    f"weights({fname}).head_or_connector",
                    has_head_key,
                    f"{'found' if has_head_key else 'MISSING'}",
                )

            # Check layer indices are present
            if layer_end > layer_start:
                for idx in (layer_start, layer_end - 1):
                    has_layer = any(f".{idx}." in k or f"_{idx}" in k for k in keys)
                    if not has_layer:
                        result.warn(
                            f"weights({fname}).layer_{idx}",
                            f"layer {idx} not found in weight keys",
                        )

        except Exception as e:
            result.check(f"weights({fname})", False, f"error loading: {e}")


def verify_vectors(manifest: dict, base_dir: str, result: VerifyResult):
    """Verify verification vectors against the first text shard."""
    vectors = manifest.get("verification_vectors", [])
    if not vectors:
        result.warn("verification_vectors", "none in manifest — skipping")
        return

    # Find first shard with an embedding (v3: check has_embedding; v2: check text_shards)
    first_shard = None
    for shard in _get_all_shards(manifest):
        if shard.get("has_embedding"):
            first_shard = shard
            break

    if first_shard is None:
        result.warn("verification_vectors", "no embedding shard found — skipping")
        return

    path = os.path.join(base_dir, first_shard["file"])
    if not os.path.isfile(path):
        result.warn("verification_vectors", f"shard file not found: {path}")
        return

    try:
        # Load embedding weights
        ext = Path(path).suffix.lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(path, device="cpu")
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)

        # Find embedding weight
        embed_weight = None
        for k, v in sd.items():
            if "embed_tokens" in k and "weight" in k:
                embed_weight = v
                break

        if embed_weight is None:
            result.warn("verification_vectors", "embed_tokens.weight not found in shard")
            return

        # Run verification vectors
        embed = torch.nn.Embedding(embed_weight.shape[0], embed_weight.shape[1])
        embed.weight = torch.nn.Parameter(embed_weight)

        for vec in vectors:
            vid = vec["vector_id"]
            input_ids = torch.tensor([vec["input_ids"]])

            with torch.no_grad():
                output = embed(input_ids)

            # Check embedding output hash
            actual_hash = hashlib.sha256(
                output.float().numpy().tobytes()
            ).hexdigest()[:16]
            expected_hash = vec.get("embed_output_hash", "")

            if expected_hash:
                match = actual_hash == expected_hash
                result.check(
                    f"verify_vector[{vid}].hash",
                    match,
                    f"{'match' if match else f'MISMATCH: expected {expected_hash}, got {actual_hash}'}",
                )

            # Check embedding output norm
            actual_norm = float(output.norm().item())
            expected_norm = vec.get("embed_output_norm", 0)
            if expected_norm:
                # Allow small numerical tolerance
                norm_match = abs(actual_norm - expected_norm) < 1e-4
                result.check(
                    f"verify_vector[{vid}].norm",
                    norm_match,
                    f"expected {expected_norm:.6f}, got {actual_norm:.6f}",
                )

    except Exception as e:
        result.check("verification_vectors", False, f"error: {e}")


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_shards(
    manifest_path: str,
    deep: bool = False,
    source_model: Optional[str] = None,
) -> VerifyResult:
    """Run all verification checks on a set of shards.

    Args:
        manifest_path: Path to manifest.json.
        deep: Run deep verification (compare with source model).
        source_model: Path to original model for deep comparison.

    Returns:
        VerifyResult with all check results.
    """
    result = VerifyResult()
    start_time = time.time()
    base_dir = str(Path(manifest_path).parent)

    print("=" * 60)
    print("  UNFED Shard Verifier")
    print("=" * 60)
    print()
    print(f"  Manifest: {manifest_path}")
    print()

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        result.check("manifest_load", False, f"error: {e}")
        return result

    result.check("manifest_load", True, "OK")
    print(f"  Model: {manifest.get('model_id', 'unknown')}")
    print(f"  Type:  {manifest.get('model_type', 'unknown')}")
    print()

    # Step 1: Structure
    print("  [1/5] Checking manifest structure...")
    verify_manifest_structure(manifest, result)

    # Step 2: Files
    print("  [2/5] Checking file existence and sizes...")
    verify_files_exist(manifest, base_dir, result)

    # Step 3: Hashes
    print("  [3/5] Verifying SHA256 hashes...")
    verify_hashes(manifest, base_dir, result)

    # Step 4: Chunk hashes
    print("  [4/5] Verifying chunk hashes...")
    verify_chunk_hashes(manifest, base_dir, result)

    # Step 5: Weight keys + verification vectors
    print("  [5/5] Verifying weight contents...")
    verify_weight_keys(manifest, base_dir, result)
    verify_vectors(manifest, base_dir, result)

    # Deep verification (optional)
    if deep and source_model:
        print()
        print("  [DEEP] Comparing with source model...")
        # Deep verification would load the original model and compare
        # weight values — deferred for now
        result.warn("deep_verify", "not yet implemented")

    elapsed = time.time() - start_time

    # Print results
    print()
    print("-" * 60)
    for check in result.checks:
        status_char = {"PASS": "+", "FAIL": "X", "WARN": "!"}[check["status"]]
        detail = f" — {check['detail']}" if check["detail"] else ""
        print(f"  [{status_char}] {check['name']}{detail}")

    print()
    print("=" * 60)
    status = "ALL CHECKS PASSED" if result.ok else f"{result.failed} CHECK(S) FAILED"
    print(f"  {status} ({result.summary()}) in {elapsed:.1f}s")
    print("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_verify(args):
    """CLI handler for 'unfed verify'."""
    result = verify_shards(
        manifest_path=args.manifest_path,
        deep=args.deep,
        source_model=args.source,
    )
    sys.exit(0 if result.ok else 1)
