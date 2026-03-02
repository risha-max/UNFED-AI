import hashlib

from economics.onchain import _to_bytes32_hash


def test_to_bytes32_hash_accepts_hex():
    h = "a" * 64
    out = _to_bytes32_hash(h)
    assert out == bytes.fromhex(h)


def test_to_bytes32_hash_accepts_prefixed_hex():
    h = "0x" + ("b" * 64)
    out = _to_bytes32_hash(h)
    assert out == bytes.fromhex("b" * 64)


def test_to_bytes32_hash_hashes_legacy_uuid_string():
    legacy = "a03d4d32-8472-43e4-8ef9-54d92c1fac18"
    out = _to_bytes32_hash(legacy)
    assert out == hashlib.sha256(legacy.encode("utf-8")).digest()
