/* ================================================================
   UNFED AI Dashboard — App Router & Shared State
   ================================================================ */

const App = {
    // Shared state between tabs
    state: {
        circuit: null,       // Current circuit info from WebSocket
        nodes: [],           // All registered nodes
        chatWs: null,        // Chat WebSocket connection
        chainWs: null,       // Chain WebSocket connection
        generating: false,   // Whether generation is in progress
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
        this.connectChat();
        this.connectChain();

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
