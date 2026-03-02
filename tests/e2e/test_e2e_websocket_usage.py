import pytest

from web import server as web_server


class _FakeNode:
    def __init__(self, address: str, shard_index: int, node_type: str = "compute"):
        self.address = address
        self.layer_start = shard_index
        self.layer_end = shard_index + 1
        self.has_embedding = shard_index == 0
        self.has_lm_head = shard_index == 1
        self.node_type = node_type
        self.shard_index = shard_index
        self.model_id = "model-x"


class _FakeModel:
    def __init__(self, model_id: str, can_serve: bool = True):
        self.model_id = model_id
        self.total_nodes = 2
        self.total_shards = 2
        self.covered_shards = 2 if can_serve else 1
        self.can_serve = can_serve


class _FakeDiscovery:
    def list_models(self):
        return [
            _FakeModel("Qwen/Qwen2.5-Coder-0.5B-Instruct"),
            _FakeModel("Qwen/Qwen2-VL-2B-Instruct"),
            _FakeModel("HuggingFaceTB/SmolVLM-256M-Instruct"),
            _FakeModel("model-x"),
        ]

    def build_circuit(self, model_id: str):
        return (["mpc-a:50049", "node-b:50052"], [b"pk-a", b"pk-b"])

    def build_vision_circuit(self, model_id: str):
        return (["vision-a:50061"], [b"pk"])

    def discover(self, model_id: str):
        nodes = [
            _FakeNode("mpc-a:50049", 0, "mpc"),
            _FakeNode("node-b:50052", 1, "compute"),
            _FakeNode("vision-a:50061", 0, "vision"),
        ]
        if not model_id:
            nodes.append(_FakeNode("daemon:50070", -1, "daemon"))
        return nodes

    def discover_compute(self, model_id: str):
        return [_FakeNode("node-b:50052", 1, "compute")]

    def discover_mpc(self, model_id: str):
        return [_FakeNode("mpc-a:50049", 0, "mpc")]

    def get_verifier_health(self):
        class _H:
            healthy_verifier_count = 1
            required_verifier_count = 1
            healthy = True
        return _H()


class _FakeDiscoveryNoMpc(_FakeDiscovery):
    def build_circuit(self, model_id: str):
        return (["node-a:50051", "node-b:50052"], [b"pk-a", b"pk-b"])

    def discover(self, model_id: str):
        nodes = [
            _FakeNode("node-a:50051", 0, "compute"),
            _FakeNode("node-b:50052", 1, "compute"),
            _FakeNode("vision-a:50061", 0, "vision"),
        ]
        if not model_id:
            nodes.append(_FakeNode("daemon:50070", -1, "daemon"))
        return nodes

    def discover_compute(self, model_id: str):
        return [
            _FakeNode("node-a:50051", 0, "compute"),
            _FakeNode("node-b:50052", 1, "compute"),
        ]

    def discover_mpc(self, model_id: str):
        return []


class _FakeDiscoveryNoDaemon(_FakeDiscovery):
    def discover(self, model_id: str):
        return [
            _FakeNode("mpc-a:50049", 0, "mpc"),
            _FakeNode("node-b:50052", 1, "compute"),
            _FakeNode("vision-a:50061", 0, "vision"),
        ]


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class _WsChatProbe:
    def __init__(self, wallet: str):
        self.query_params = {"wallet": wallet}
        self.messages = []
        self.closed_code = None

    async def accept(self):
        return None

    async def send_json(self, payload):
        self.messages.append(payload)

    async def close(self, code: int = 1000):
        self.closed_code = code

    async def receive_json(self):
        raise web_server.WebSocketDisconnect()


@pytest.mark.anyio
async def test_ws_usage_text_exact_tokens_and_streaming_consistency(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, prompt: str, model_id: str = None):
            return 7

        def count_qwen2_vl_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def count_smolvlm_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def generate(self, *args, **kwargs):
            for token in ["A", "B", "C"]:
                yield token

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    captured = {}

    def fake_report_usage(**kwargs):
        captured.update(kwargs)
        return 0.321

    monkeypatch.setattr(web_server, "_report_usage", fake_report_usage)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello world",
        image_path=None,
        model_type="qwen2",
        max_tokens=8,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )

    done = [m for m in ws.messages if m.get("type") == "done"][-1]
    token_msgs = [m for m in ws.messages if m.get("type") == "token"]
    assert done["input_tokens"] == 7
    assert done["output_tokens"] == len(token_msgs) == 3
    assert done["total_tokens"] == len(token_msgs)
    assert done["cost"] == pytest.approx(0.321, rel=1e-9)
    assert captured["input_tokens"] == 7
    assert captured["output_tokens"] == 3


@pytest.mark.anyio
async def test_ws_usage_qwen2_vl_exact_tokens(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def count_qwen2_vl_input_tokens(self, *args, **kwargs):
            return 33

        def count_smolvlm_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def generate_multimodal(self, *args, **kwargs):
            for token in ["vision", "ok"]:
                yield token

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    monkeypatch.setattr(web_server, "_report_usage", lambda **_: 0.5)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="What is in this image?",
        image_path="/tmp/fake.png",
        model_type="qwen2_vl",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )

    done = [m for m in ws.messages if m.get("type") == "done"][-1]
    assert done["input_tokens"] == 33
    assert done["output_tokens"] == 2


@pytest.mark.anyio
async def test_ws_usage_smolvlm_exact_tokens(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def count_qwen2_vl_input_tokens(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def count_smolvlm_input_tokens(self, *args, **kwargs):
            return 29

        def generate_multimodal_smolvlm(self, *args, **kwargs):
            for token in ["smol", "vlm", "ok"]:
                yield token

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    monkeypatch.setattr(web_server, "_report_usage", lambda **_: 0.75)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="Describe this image",
        image_path="/tmp/fake.png",
        model_type="smolvlm",
        max_tokens=4,
        use_voting=False,
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )
    done = [m for m in ws.messages if m.get("type") == "done"][-1]
    assert done["input_tokens"] == 29
    assert done["output_tokens"] == 3


@pytest.mark.anyio
async def test_ws_usage_strict_failure_on_tokenizer_unavailable(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, *args, **kwargs):
            raise RuntimeError("Tokenizer unavailable for selected model_id")

        def count_qwen2_vl_input_tokens(self, *args, **kwargs):
            return 0

        def count_smolvlm_input_tokens(self, *args, **kwargs):
            return 0

        def generate(self, *args, **kwargs):
            raise AssertionError("generation should not run")

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    called = {"count": 0}

    def fake_report_usage(**kwargs):
        called["count"] += 1
        return 0.0

    monkeypatch.setattr(web_server, "_report_usage", fake_report_usage)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello",
        image_path=None,
        model_type="qwen2",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )

    assert called["count"] == 0
    assert any(
        m.get("type") == "error" and "Tokenizer unavailable" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_usage_rejects_adversarial_token_count(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, *args, **kwargs):
            return 9999

        def count_qwen2_vl_input_tokens(self, *args, **kwargs):
            return 0

        def count_smolvlm_input_tokens(self, *args, **kwargs):
            return 0

        def generate(self, *args, **kwargs):
            for token in ["A"]:
                yield token

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    monkeypatch.setattr(web_server, "_MAX_BILLING_INPUT_TOKENS", 8)
    called = {"count": 0}

    def fake_report_usage(**kwargs):
        called["count"] += 1
        return 0.0

    monkeypatch.setattr(web_server, "_report_usage", fake_report_usage)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="x" * 10000,
        image_path=None,
        model_type="qwen2",
        max_tokens=2,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )

    assert called["count"] == 0
    assert any(
        m.get("type") == "error" and "Invalid input token count" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_usage_rejects_disallowed_cluster_endpoint(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("client must not initialize for disallowed endpoint")

    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello",
        image_path=None,
        model_type="qwen2",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="evil.example:50050",
        client_address="0xabc",
    )
    assert any(
        m.get("type") == "error" and "cluster_endpoint is not allowed" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_chat_rejects_invalid_wallet_address():
    ws = _WsChatProbe("not-an-address")
    await web_server.ws_chat(ws)
    assert ws.closed_code == 1008
    assert any(
        m.get("type") == "error" and "Invalid wallet address format" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_usage_rejects_when_mpc_required_but_absent(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscoveryNoMpc())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello",
        image_path=None,
        model_type="qwen2",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )
    assert any(
        m.get("type") == "error" and "missing MPC shard-0 entry" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_usage_rejects_when_daemon_required_but_absent(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.delenv("UNFED_REQUIRE_DAEMON", raising=False)
    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscoveryNoDaemon())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello",
        image_path=None,
        model_type="qwen2",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )
    assert any(
        m.get("type") == "error" and "No healthy daemon available" in m.get("message", "")
        for m in ws.messages
    )


@pytest.mark.anyio
async def test_ws_usage_allows_non_mpc_when_override_set(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def count_text_input_tokens(self, prompt: str, model_id: str = None):
            return 3

        def generate(self, *args, **kwargs):
            for token in ["x"]:
                yield token

    monkeypatch.setenv("UNFED_REQUIRE_MPC", "0")
    monkeypatch.setattr("client.client.UnfedClient", FakeClient)
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscoveryNoMpc())
    monkeypatch.setattr(web_server, "_check_client_balance", lambda *_: (True, 10.0))
    monkeypatch.setattr(web_server, "_report_usage", lambda **_: 0.0)

    ws = _FakeWebSocket()
    await web_server._run_generation(
        websocket=ws,
        prompt="hello",
        image_path=None,
        model_type="qwen2",
        max_tokens=4,
        use_voting=False,
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        cluster_endpoint="",
        client_address="0xabc",
    )
    assert any(m.get("type") == "done" for m in ws.messages)
