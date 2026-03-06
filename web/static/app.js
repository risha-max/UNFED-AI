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
        authSessionToken: "",   // Wallet auth session token for chat
        activeWallet: "",       // Wallet resolved by backend session
        devAuthBypass: false,   // Local dev bypass mode from backend
        heOutputEnabled: false, // HE output artifact mode capability
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
        const buttons = document.querySelectorAll('.topbar-tabs .tab-btn');
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
        const authBtn = document.getElementById('walletAuthBtn');

        // Restore from localStorage
        const saved = localStorage.getItem('unfed_wallet');
        if (saved) el.value = saved;
        const savedSession = localStorage.getItem('unfed_wallet_session');
        if (savedSession && !this.state.devAuthBypass) {
            this.state.authSessionToken = savedSession;
            this.setWalletHint('Session restored. Reconnecting...');
        } else if (this.state.devAuthBypass) {
            this.setWalletHint('Dev mode: wallet field is trusted for local testing.');
        } else {
            this.setWalletHint('Sign in with a wallet signature before chat.');
        }

        // Persist on change and force re-auth for this wallet.
        el.addEventListener('change', () => {
            const addr = el.value.trim();
            if (addr) {
                localStorage.setItem('unfed_wallet', addr);
            } else {
                localStorage.removeItem('unfed_wallet');
            }
            if (!this.state.devAuthBypass) {
                this.clearWalletSession('Wallet changed. Please sign in again.');
            }
            if (this.state.chatWs) {
                this.state.chatWs.close();
            } else if (this.state.devAuthBypass && addr) {
                this.connectChat();
            }
        });

        if (authBtn) {
            authBtn.addEventListener('click', () => this.authenticateWallet());
        }
    },

    applyAuthModeUi() {
        const authBtn = document.getElementById('walletAuthBtn');
        if (this.state.devAuthBypass) {
            if (authBtn) authBtn.style.display = 'none';
            this.clearWalletSession();
            this.setWalletHint('Dev mode: wallet field is trusted for local testing.');
            return;
        }
        if (authBtn) authBtn.style.display = '';
        const savedSession = localStorage.getItem('unfed_wallet_session');
        if (savedSession) {
            this.state.authSessionToken = savedSession;
            this.setWalletHint('Session restored. Reconnecting...');
        } else {
            this.setWalletHint('Sign in with a wallet signature before chat.');
        }
    },

    async loadAuthMode() {
        const mode = await this.fetchJson('/api/client/auth/mode');
        this.state.devAuthBypass = Boolean(mode && mode.dev_auth_bypass);
        this.applyAuthModeUi();
    },

    async loadSecurityModes() {
        const mode = await this.fetchJson('/api/security/modes');
        this.state.heOutputEnabled = Boolean(mode && mode.he_output_enabled);
    },

    setWalletHint(message, isError = false) {
        const hint = document.getElementById('walletHint');
        if (!hint) return;
        hint.textContent = message;
        hint.style.color = isError ? '#ef4444' : '';
    },

    clearWalletSession(message = '') {
        this.state.authSessionToken = '';
        this.state.activeWallet = '';
        localStorage.removeItem('unfed_wallet_session');
        if (message) this.setWalletHint(message, true);
    },

    async authenticateWallet() {
        const authBtn = document.getElementById('walletAuthBtn');
        if (!window.ethereum) {
            this.setWalletHint('No wallet provider found. Install MetaMask (or compatible).', true);
            return;
        }

        if (authBtn) authBtn.disabled = true;
        this.setWalletHint('Requesting wallet signature...');
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            const signerAddress = (accounts && accounts[0]) ? String(accounts[0]) : '';
            if (!signerAddress) throw new Error('No wallet account available');

            const entered = this.getWalletAddress();
            if (entered && entered.toLowerCase() !== signerAddress.toLowerCase()) {
                // Prefer the actual signer account; users often have stale/typed
                // addresses in the field while MetaMask is connected to another one.
                this.setWalletHint('Using connected signer account from wallet extension.');
            }

            const walletEl = document.getElementById('walletAddress');
            if (walletEl) walletEl.value = signerAddress;
            localStorage.setItem('unfed_wallet', signerAddress);

            const challengeResp = await this.postJson('/api/client/auth', {});
            if (!challengeResp || !challengeResp.challenge) {
                throw new Error('Failed to fetch auth challenge');
            }

            const signature = await window.ethereum.request({
                method: 'personal_sign',
                params: [challengeResp.challenge, signerAddress],
            });

            const verifyResp = await this.postJson('/api/client/auth/verify', {
                challenge: challengeResp.challenge,
                signature,
                address: signerAddress,
            });
            if (!verifyResp || !verifyResp.success || !verifyResp.session_token) {
                throw new Error(verifyResp?.error || 'Wallet verification failed');
            }

            this.state.authSessionToken = verifyResp.session_token;
            this.state.activeWallet = verifyResp.address || signerAddress;
            localStorage.setItem('unfed_wallet_session', verifyResp.session_token);
            this.setWalletHint(`Signed in as ${this.state.activeWallet.slice(0, 6)}...${this.state.activeWallet.slice(-4)}`);

            if (this.state.chatWs) {
                this.state.chatWs.close();
            } else {
                this.connectChat();
            }
        } catch (err) {
            const msg = err?.message || String(err);
            this.clearWalletSession(`Wallet sign-in failed: ${msg}`);
        } finally {
            if (authBtn) authBtn.disabled = false;
        }
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
        let qs = '';
        const sessionToken = this.state.authSessionToken || localStorage.getItem('unfed_wallet_session') || '';
        if (this.state.devAuthBypass) {
            const wallet = this.getWalletAddress();
            if (!wallet) {
                this.setStatus('error', 'Enter wallet address');
                return;
            }
            qs = `?wallet=${encodeURIComponent(wallet)}`;
        } else {
            if (!sessionToken) {
                this.setStatus('error', 'Wallet sign-in required');
                return;
            }
            qs = `?session=${encodeURIComponent(sessionToken)}`;
        }
        const ws = new WebSocket(`${protocol}//${location.host}/ws/chat${qs}`);

        ws.onopen = () => {
            if (!this.state.devAuthBypass) {
                this.state.authSessionToken = sessionToken;
            }
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
                if (msg.type === 'error' && String(msg.message || '').toLowerCase().includes('wallet authentication required')) {
                    this.clearWalletSession('Wallet session expired. Please sign in again.');
                    this.setStatus('error', 'Wallet sign-in required');
                    return;
                }
                this.emit('chatMessage', msg);
            } catch (e) {
                console.error('Chat WS parse error:', e);
            }
        };

        ws.onclose = (event) => {
            this.state.chatWs = null;
            const unauthorized = event && event.code === 1008;
            if (unauthorized) {
                if (!this.state.devAuthBypass) {
                    this.clearWalletSession('Wallet session expired. Please sign in again.');
                    this.setStatus('error', 'Wallet sign-in required');
                } else {
                    this.setStatus('error', 'Wallet rejected');
                }
                return;
            }
            this.setStatus('error', 'Disconnected');
            if (this.state.authSessionToken) {
                // Reconnect after 3s while session remains valid.
                setTimeout(() => this.connectChat(), 3000);
            }
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

    async postJson(url, body) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body || {}),
            });
            return await res.json();
        } catch (e) {
            console.error('POST error:', url, e);
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
        Promise.all([this.loadAuthMode(), this.loadSecurityModes()]).finally(() => this.connectChat());
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
