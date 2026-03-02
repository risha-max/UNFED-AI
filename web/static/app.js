/* ================================================================
   UNFED AI Dashboard — App Router & Shared State
   ================================================================ */

const App = {
    // Shared state between tabs
    state: {
        circuit: null,          // Current circuit info from WebSocket
        nodes: [],              // All registered nodes
        chatWs: null,           // Chat WebSocket connection
        chainWs: null,          // Chain WebSocket connection
        generating: false,      // Whether generation is in progress
        models: [],             // Resolved model catalog
        selectedModelId: "",    // Active model_id
    },

    // Event bus for cross-tab communication
    listeners: {},

    on(event, fn) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(fn);
    },

    emit(event, data) {
        (this.listeners[event] || []).forEach(fn => fn(data));
    },

    // ---- Tab routing ----
    initTabs() {
        const buttons = document.querySelectorAll('.tab-btn');
        const contents = document.querySelectorAll('.tab-content');

        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;

                buttons.forEach(b => b.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                document.getElementById('tab-' + tab).classList.add('active');

                this.emit('tabSwitch', tab);
            });
        });
    },

    // ---- Wallet ----
    getWalletAddress() {
        const el = document.getElementById('walletAddress');
        return el ? el.value.trim() : '';
    },

    initWallet() {
        const el = document.getElementById('walletAddress');
        if (!el) return;

        // Restore from localStorage
        const saved = localStorage.getItem('unfed_wallet');
        if (saved) el.value = saved;

        // Persist on change and reconnect WebSocket so the new address takes effect
        el.addEventListener('change', () => {
            const addr = el.value.trim();
            if (addr) {
                localStorage.setItem('unfed_wallet', addr);
            } else {
                localStorage.removeItem('unfed_wallet');
            }
            // Reconnect with the new wallet
            if (this.state.chatWs) {
                this.state.chatWs.close();
            }
        });
    },

    // ---- Model mode / selector ----
    inferModelType(modelId) {
        const lower = String(modelId || "").toLowerCase();
        if (lower.includes("smolvlm")) return "smolvlm";
        if (lower.includes("qwen2-vl") || lower.includes("qwen2_vl")) return "qwen2_vl";
        return "qwen2";
    },

    isVisionCapable(modelType, modelId) {
        if (modelType === "smolvlm" || modelType === "qwen2_vl") {
            return true;
        }
        const lower = String(modelId || "").toLowerCase();
        return lower.includes("vl") || lower.includes("vision");
    },

    normalizeRegistryModels(rawModels) {
        return (rawModels || []).map((m) => {
            const modelId = m.model_id || "";
            const modelType = this.inferModelType(modelId);
            const totalShards = Number(m.total_shards || 0);
            const coveredShards = Number(m.covered_shards || 0);
            const canServe = Boolean(
                m.can_serve !== undefined ? m.can_serve : m.is_healthy
            );
            return {
                model_id: modelId,
                model_type: modelType,
                vision_capable: this.isVisionCapable(modelType, modelId),
                total_nodes: Number(m.total_nodes || 0),
                total_shards: totalShards,
                covered_shards: coveredShards,
                can_serve: canServe,
                label: `${modelId} (${coveredShards}/${totalShards} shards)`,
            };
        });
    },

    setModelSourceHint(text, isError = false) {
        const hint = document.getElementById("modelSourceHint");
        if (!hint) return;
        hint.textContent = text;
        hint.style.color = isError ? "#ef4444" : "";
    },

    populateModelSelect(models, placeholder) {
        const select = document.getElementById("modelSelect");
        if (!select) return;
        select.innerHTML = "";

        if (!models.length) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = placeholder;
            opt.disabled = true;
            opt.selected = true;
            select.appendChild(opt);
            this.state.models = [];
            this.state.selectedModelId = "";
            this.emit("modelSelectionChanged", null);
            return;
        }

        const preferred = localStorage.getItem("unfed_model_registry") || "";
        let selectedId = "";

        models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.model_id;
            opt.dataset.modelType = m.model_type;
            opt.dataset.vision = m.vision_capable ? "true" : "false";
            opt.dataset.canServe = m.can_serve ? "true" : "false";
            opt.disabled = !m.can_serve;
            opt.textContent = m.can_serve ? m.label : `${m.label} — unavailable`;
            select.appendChild(opt);
        });

        if (preferred && models.some((m) => m.model_id === preferred && m.can_serve)) {
            selectedId = preferred;
        } else {
            const firstHealthy = models.find((m) => m.can_serve);
            selectedId = firstHealthy ? firstHealthy.model_id : "";
        }

        if (selectedId) {
            select.value = selectedId;
            this.state.selectedModelId = selectedId;
            localStorage.setItem("unfed_model_registry", selectedId);
            this.emit("modelSelectionChanged", this.getSelectedModel());
            return;
        }

        this.state.selectedModelId = "";
        this.emit("modelSelectionChanged", null);
    },

    getSelectedModel() {
        const select = document.getElementById("modelSelect");
        if (!select || !select.value) return null;
        const modelId = select.value;
        const modelType = select.selectedOptions[0]?.dataset?.modelType || this.inferModelType(modelId);
        const visionCapable = select.selectedOptions[0]?.dataset?.vision === "true";
        const canServe = select.selectedOptions[0]?.dataset?.canServe !== "false";
        return {
            model_id: modelId,
            model_type: modelType,
            vision_capable: visionCapable,
            can_serve: canServe,
        };
    },

    async refreshModelOptions() {
        const data = await this.fetchJson("/api/models");
        if (!data || data.error) {
            this.populateModelSelect([], "Failed to load models from registry");
            this.setModelSourceHint("Model discovery failed from registry.", true);
            return;
        }
        const models = this.normalizeRegistryModels(data.models);
        this.state.models = models;
        this.populateModelSelect(models, "No models available in registry");
        this.setModelSourceHint("Live model catalog from this registry.");
    },

    initModelMode() {
        const select = document.getElementById("modelSelect");
        if (!select) return;

        select.addEventListener("change", () => {
            const selected = this.getSelectedModel();
            if (!selected) return;
            this.state.selectedModelId = selected.model_id;
            localStorage.setItem("unfed_model_registry", selected.model_id);
            this.emit("modelSelectionChanged", selected);
        });
    },

    // ---- WebSocket: Chat ----
    connectChat() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wallet = this.getWalletAddress();
        const qs = wallet ? `?wallet=${encodeURIComponent(wallet)}` : '';
        const ws = new WebSocket(`${protocol}//${location.host}/ws/chat${qs}`);

        ws.onopen = () => {
            this.state.chatWs = ws;
            this.setStatus('connected', 'Connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                // Server confirms the resolved wallet address
                if (msg.type === 'wallet') {
                    this.state.activeWallet = msg.address;
                    const hint = document.getElementById('walletHint');
                    if (hint) {
                        const short = msg.address.slice(0, 6) + '...' + msg.address.slice(-4);
                        hint.textContent = `Active: ${short}`;
                    }
                    return;
                }
                this.emit('chatMessage', msg);
            } catch (e) {
                console.error('Chat WS parse error:', e);
            }
        };

        ws.onclose = () => {
            this.state.chatWs = null;
            this.setStatus('error', 'Disconnected');
            // Reconnect after 3s
            setTimeout(() => this.connectChat(), 3000);
        };

        ws.onerror = (err) => {
            console.error('Chat WS error:', err);
            this.setStatus('error', 'Connection error');
        };
    },

    // ---- WebSocket: Chain ----
    connectChain() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${location.host}/ws/chain`);

        ws.onopen = () => {
            this.state.chainWs = ws;
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this.emit('chainMessage', msg);
            } catch (e) {
                console.error('Chain WS parse error:', e);
            }
        };

        ws.onclose = () => {
            this.state.chainWs = null;
            setTimeout(() => this.connectChain(), 5000);
        };

        ws.onerror = () => {};
    },

    // ---- Send chat message ----
    sendChatMessage(data) {
        if (this.state.chatWs && this.state.chatWs.readyState === WebSocket.OPEN) {
            this.state.chatWs.send(JSON.stringify(data));
            this.state.generating = true;
            return true;
        }
        return false;
    },

    // ---- Status indicator ----
    setStatus(state, text) {
        const dot = document.getElementById('statusDot');
        const txt = document.getElementById('statusText');
        dot.className = 'status-dot ' + state;
        txt.textContent = text;
    },

    // ---- Fetch helpers ----
    async fetchJson(url) {
        try {
            const res = await fetch(url);
            return await res.json();
        } catch (e) {
            console.error('Fetch error:', url, e);
            return null;
        }
    },

    async postFile(url, file) {
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch(url, { method: 'POST', body: form });
            return await res.json();
        } catch (e) {
            console.error('Upload error:', e);
            return null;
        }
    },

    // ---- Format helpers ----
    truncHash(hash, len = 8) {
        if (!hash) return '—';
        return hash.length > len * 2
            ? hash.slice(0, len) + '...' + hash.slice(-len)
            : hash;
    },

    formatTime(ts) {
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString();
    },

    formatDate(ts) {
        const d = new Date(ts * 1000);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
    },

    // ---- Faucet ----
    initFaucet() {
        const btn = document.getElementById('faucetBtn');
        if (!btn) return;
        btn.addEventListener('click', () => this.requestFaucet());
    },

    async requestFaucet() {
        const address = this.getWalletAddress();
        if (!address) {
            this.setFaucetStatus('Enter a wallet address first.', true);
            return;
        }
        const btn = document.getElementById('faucetBtn');
        btn.disabled = true;
        this.setFaucetStatus('Requesting tokens...', false);

        try {
            const res = await fetch('/api/faucet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ address }),
            });
            const data = await res.json();
            if (res.ok && data.success) {
                this.setFaucetStatus(
                    `Received ${data.amount} tokens. Balance: ${data.balance}`,
                    false);
            } else if (res.status === 429) {
                const mins = Math.ceil((data.retry_after_seconds || 0) / 60);
                this.setFaucetStatus(
                    `Cooldown active. Try again in ${mins} min.`, true);
            } else {
                this.setFaucetStatus(data.error || 'Faucet error.', true);
            }
        } catch (e) {
            this.setFaucetStatus('Network error.', true);
        } finally {
            btn.disabled = false;
        }
    },

    setFaucetStatus(msg, isError) {
        const el = document.getElementById('faucetStatus');
        if (!el) return;
        el.textContent = msg;
        el.style.color = isError ? '#ef4444' : '#4ade80';
    },

    // ---- Init ----
    init() {
        this.initTabs();
        this.initWallet();
        this.initFaucet();
        this.initModelMode();
        this.connectChat();
        this.connectChain();
        this.refreshModelOptions();

        // Check registry health
        this.fetchJson('/api/network/nodes').then(data => {
            if (data && data.nodes) {
                this.state.nodes = data.nodes;
                this.setStatus('connected', `${data.nodes.length} nodes`);
                this.emit('nodesLoaded', data.nodes);
            }
        });

    },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
