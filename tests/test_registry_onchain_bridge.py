from network.registry_server import _is_eth_address


def test_is_eth_address_accepts_valid():
    assert _is_eth_address("0x5FbDB2315678afecb367f032d93F642f64180aa3")


def test_is_eth_address_rejects_uuid():
    assert not _is_eth_address("a03d4d32-8472-43e4-8ef9-54d92c1fac18")
