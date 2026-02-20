"""E2E tests for the testnet token faucet."""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import requests

from tests.e2e.conftest import PROJECT_ROOT, PYTHON

PYTHON = os.environ.get("UNFED_E2E_PYTHON", sys.executable)
ANVIL_PORT = int(os.environ.get("UNFED_E2E_ANVIL_PORT", "18545"))
WEB_PORT = int(os.environ.get("UNFED_E2E_WEB_PORT", "18080"))
REGISTRY_PORT = int(os.environ.get("UNFED_E2E_FAUCET_REGISTRY_PORT", "51150"))

ANVIL_KEY_0 = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
ANVIL_ADDR_0 = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
CLIENT_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

RPC_URL = f"http://127.0.0.1:{ANVIL_PORT}"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


def _has_foundry() -> bool:
    try:
        subprocess.run(["anvil", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_foundry = pytest.mark.skipif(
    not _has_foundry(), reason="Foundry (anvil/forge) not installed")


@pytest.fixture(scope="module")
def anvil_chain():
    """Start Anvil, deploy contracts, yield config dict, then tear down."""
    proc = subprocess.Popen(
        ["anvil", "--port", str(ANVIL_PORT), "--accounts", "10", "--silent"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if not _wait_for_port(ANVIL_PORT, timeout=10):
        proc.kill()
        pytest.skip("Anvil failed to start")

    contracts_dir = os.path.join(PROJECT_ROOT, "contracts")
    env = {
        **os.environ,
        "PATH": os.path.expanduser("~/.foundry/bin") + ":" + os.environ["PATH"],
    }
    deploy = subprocess.run(
        ["forge", "script", "script/Deploy.s.sol",
         "--rpc-url", RPC_URL, "--broadcast"],
        cwd=contracts_dir, env=env,
        capture_output=True, text=True, timeout=60,
    )
    if deploy.returncode != 0:
        proc.kill()
        pytest.skip(f"Contract deploy failed: {deploy.stderr}")

    output = deploy.stdout + deploy.stderr
    token_addr = escrow_addr = ""
    for line in output.splitlines():
        if "TOKEN_ADDRESS=" in line:
            token_addr = line.split("TOKEN_ADDRESS=")[-1].strip()
        if "ESCROW_ADDRESS=" in line:
            escrow_addr = line.split("ESCROW_ADDRESS=")[-1].strip()

    if not token_addr or not escrow_addr:
        proc.kill()
        pytest.skip(f"Could not parse addresses from deploy output")

    yield {
        "rpc_url": RPC_URL,
        "token_address": token_addr,
        "escrow_address": escrow_addr,
        "operator_key": ANVIL_KEY_0,
        "operator_address": ANVIL_ADDR_0,
    }

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def escrow(anvil_chain):
    """Return an OnChainEscrow connected to the local Anvil chain."""
    from economics.onchain import OnChainEscrow
    return OnChainEscrow(
        rpc_url=anvil_chain["rpc_url"],
        contract_address=anvil_chain["escrow_address"],
        operator_private_key=anvil_chain["operator_key"],
        token_address=anvil_chain["token_address"],
    )


@needs_foundry
@pytest.mark.e2e
class TestDepositForContract:

    def test_depositFor_credits_client(self, escrow, anvil_chain):
        """Operator calls depositFor(client, amount) — client balance should increase."""
        before = escrow.get_client_balance(CLIENT_ADDR)
        amount_tokens = 50
        escrow.faucet_drip(CLIENT_ADDR, amount_tokens)
        after = escrow.get_client_balance(CLIENT_ADDR)
        assert after - before == amount_tokens * 10**18


@needs_foundry
@pytest.mark.e2e
class TestFaucetEndpoint:

    @pytest.fixture(scope="class")
    def web_server(self, anvil_chain):
        """Start the web server with escrow config pointing at local Anvil."""
        env_file = os.path.join(PROJECT_ROOT, "deployed.env")
        with open(env_file, "w") as f:
            f.write(f"CHAIN_RPC_URL={anvil_chain['rpc_url']}\n")
            f.write(f"TOKEN_ADDRESS={anvil_chain['token_address']}\n")
            f.write(f"ESCROW_ADDRESS={anvil_chain['escrow_address']}\n")
            f.write(f"OPERATOR_PRIVATE_KEY={anvil_chain['operator_key']}\n")
            f.write(f"OPERATOR_ADDRESS={anvil_chain['operator_address']}\n")

        proc = subprocess.Popen(
            [PYTHON, "-m", "web.server",
             "--port", str(WEB_PORT),
             "--registry", f"localhost:{REGISTRY_PORT}"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if not _wait_for_port(WEB_PORT, timeout=15):
            proc.kill()
            pytest.skip("Web server did not start in time")

        yield f"http://127.0.0.1:{WEB_PORT}"

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        try:
            os.remove(env_file)
        except OSError:
            pass

    def test_faucet_drip_credits_client(self, web_server):
        resp = requests.post(
            f"{web_server}/api/faucet",
            json={"address": CLIENT_ADDR},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["amount"] == 100
        assert "tx_hash" in data
        assert data["balance"] > 0

    def test_faucet_cooldown_enforced(self, web_server):
        addr = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
        resp1 = requests.post(
            f"{web_server}/api/faucet",
            json={"address": addr},
            timeout=30,
        )
        assert resp1.status_code == 200

        resp2 = requests.post(
            f"{web_server}/api/faucet",
            json={"address": addr},
            timeout=30,
        )
        assert resp2.status_code == 429
        data = resp2.json()
        assert "retry_after_seconds" in data

    def test_faucet_without_escrow_returns_error(self):
        """Hitting a server without escrow config should return 503."""
        env_file = os.path.join(PROJECT_ROOT, "deployed.env")
        backup = env_file + ".bak"
        had_env = os.path.exists(env_file)
        if had_env:
            os.rename(env_file, backup)

        proc = subprocess.Popen(
            [PYTHON, "-m", "web.server",
             "--port", str(WEB_PORT + 1),
             "--registry", f"localhost:{REGISTRY_PORT}"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            if not _wait_for_port(WEB_PORT + 1, timeout=15):
                pytest.skip("Web server (no-escrow) did not start")

            resp = requests.post(
                f"http://127.0.0.1:{WEB_PORT + 1}/api/faucet",
                json={"address": CLIENT_ADDR},
                timeout=10,
            )
            assert resp.status_code == 503
            assert "error" in resp.json()
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if had_env:
                os.rename(backup, env_file)
