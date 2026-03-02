import os
import socket
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "manual_setup_mpc_testnet.sh"


def _base_env():
    env = os.environ.copy()
    env["UNFED_SKIP_PREREQ"] = "1"
    env["OPERATOR_PRIVATE_KEY"] = "0xabc"
    env["OPERATOR_ADDRESS"] = "0xdef"
    return env


def test_manual_setup_script_smoke_dry_run_commands():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run", "--yes-reuse-ports"],
        cwd=PROJECT_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "start_local_chain.sh" in result.stdout
    assert "network.registry_server" in result.stdout
    assert "node.server --shard-index 0" in result.stdout
    assert "web.server --port 8080" in result.stdout


def test_manual_setup_script_rejects_port_conflict_without_override():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 50050))
        sock.listen(1)
        result = subprocess.run(
            [str(SCRIPT), "--dry-run"],
            cwd=PROJECT_ROOT,
            env=_base_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0
    assert "Ports already in use" in result.stdout
