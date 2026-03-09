/* ================================================================
   UNFED AI Dashboard — Network Visualizer (Tor-style)
   ================================================================ */

const Network = {
    circuitContainer: null,
    textCircuitPath: null,
    visionCircuitPath: null,
    visionSection: null,
    textSection: null,
    circuitHint: null,
    nodesList: null,
    refreshBtn: null,
    modelFilterEl: null,
    typeFilterEl: null,
    healthyOnlyEl: null,
    windowEl: null,
    perfHintEl: null,

    // Track circuit node elements for hop animation
    circuitNodeEls: {},  // address -> DOM element
    circuitArrowEls: {}, // "from->to" -> DOM element

    init() {
        this.circuitContainer = document.getElementById('circuitContainer');
        this.textCircuitPath = document.getElementById('textCircuitPath');
        this.visionCircuitPath = document.getElementById('visionCircuitPath');
        this.visionSection = document.getElementById('visionCircuitSection');
        this.textSection = document.getElementById('textCircuitSection');
        this.circuitHint = document.getElementById('circuitHint');
        this.nodesList = document.getElementById('nodesList');
        this.refreshBtn = document.getElementById('refreshNodesBtn');
        this.modelFilterEl = document.getElementById('networkModelFilter');
        this.typeFilterEl = document.getElementById('networkTypeFilter');
        this.healthyOnlyEl = document.getElementById('networkHealthyOnly');
        this.windowEl = document.getElementById('registryWindow');
        this.perfHintEl = document.getElementById('registryPerfHint');

        this.refreshBtn.addEventListener('click', () => this.loadNodes());
        this.modelFilterEl?.addEventListener('change', () => this.renderNodes(App.state.nodes));
        this.typeFilterEl?.addEventListener('change', () => this.renderNodes(App.state.nodes));
        this.healthyOnlyEl?.addEventListener('change', () => this.renderNodes(App.state.nodes));

        // Listen for circuit updates from chat
        App.on('circuitUpdate', (msg) => this.renderCircuit(msg));
        App.on('hopUpdate', (msg) => this.animateHop(msg));
        App.on('nodesLoaded', (nodes) => this.renderNodes(nodes));
        App.on('registrySummaryLoaded', (summary) => {
            App.state.registrySummary = summary;
            this.renderSummary(summary);
            this.syncModelFilterOptions(summary.models || []);
            this.renderNodes(App.state.nodes || []);
        });
        App.on('modelSelectionChanged', () => this.loadRegistrySummary());
        App.on('tabSwitch', (tab) => {
            if (tab === 'network') {
                this.loadNodes();
            }
        });

        // Initial load comes from shared app bootstrap.
        if (App.state.registrySummary) {
            this.renderSummary(App.state.registrySummary);
            this.syncModelFilterOptions(App.state.registrySummary.models || []);
        }
        if (Array.isArray(App.state.nodes) && App.state.nodes.length > 0) {
            this.renderNodes(App.state.nodes);
        }
        if (typeof App.loadNetworkBootstrap === 'function') {
            App.loadNetworkBootstrap();
        } else {
            this.loadNodes();
        }
    },

    async loadNodes() {
        await this.loadRegistrySummary();
        const data = await App.fetchJson('/api/network/nodes');
        if (data && data.nodes) {
            App.state.nodes = data.nodes;
            this.renderNodes(data.nodes);
        }
    },

    selectedModelId() {
        const selected = App.getSelectedModel();
        if (selected && selected.model_id) return selected.model_id;
        return '';
    },

    async loadRegistrySummary() {
        const modelId = this.selectedModelId();
        const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
        const summary = await App.fetchJson(`/api/registry/summary${query}`);
        if (!summary || summary.error) return;
        App.state.registrySummary = summary;
        this.renderSummary(summary);
        this.syncModelFilterOptions(summary.models || []);
        this.renderNodes(App.state.nodes || []);
    },

    syncModelFilterOptions(models) {
        if (!this.modelFilterEl) return;
        const current = this.modelFilterEl.value;
        this.modelFilterEl.innerHTML = '<option value="">All models</option>';
        models
            .slice()
            .sort((a, b) => String(a.model_id || '').localeCompare(String(b.model_id || '')))
            .forEach((m) => {
                const opt = document.createElement('option');
                opt.value = m.model_id || '';
                opt.textContent = m.can_serve
                    ? `${m.model_id} (${m.covered_shards}/${m.total_shards})`
                    : `${m.model_id} (${m.covered_shards}/${m.total_shards}) - unavailable`;
                this.modelFilterEl.appendChild(opt);
            });
        if (current && models.some((m) => m.model_id === current)) {
            this.modelFilterEl.value = current;
        } else if (!current && this.selectedModelId()) {
            this.modelFilterEl.value = this.selectedModelId();
        }
    },

    formatRate(value, suffix = '') {
        const num = Number(value || 0);
        if (!Number.isFinite(num) || num <= 0) return '—';
        return `${num.toFixed(2)}${suffix}`;
    },

    formatMs(value) {
        const num = Number(value || 0);
        if (!Number.isFinite(num) || num <= 0) return '—';
        return `${Math.round(num)}ms`;
    },

    renderSummary(summary) {
        const perf = summary.performance || {};
        const pricing = summary.pricing || {};
        const registry = summary.registry || {};

        const currency = pricing.currency || 'UNFED';
        document.getElementById('regTotalNodes').textContent = String(registry.total_nodes ?? '—');
        document.getElementById('regPriceIn').textContent =
            Number(pricing.price_per_input_token || 0) > 0
                ? `${Number(pricing.price_per_input_token).toFixed(6)} ${currency}`
                : '—';
        document.getElementById('regPriceOut').textContent =
            Number(pricing.price_per_output_token || 0) > 0
                ? `${Number(pricing.price_per_output_token).toFixed(6)} ${currency}`
                : '—';
        document.getElementById('regAvgTps').textContent = this.formatRate(perf.avg_tps, ' tok/s');
        document.getElementById('regP95Ttft').textContent = this.formatMs(perf.p95_ttft_ms);
        document.getElementById('regHealthyModels').textContent =
            `${registry.healthy_models ?? 0}/${registry.total_models ?? 0}`;

        if (this.windowEl) {
            const seconds = Number(perf.window_seconds || 0);
            this.windowEl.textContent = seconds > 0
                ? `Window: last ${Math.round(seconds / 60)}m`
                : 'Window: --';
        }
        if (this.perfHintEl) {
            const samples = Number(perf.sample_count || 0);
            this.perfHintEl.textContent = samples > 0
                ? `Performance values are rolling end-to-end user metrics (${samples} sample${samples === 1 ? '' : 's'}).`
                : 'Performance values will appear after completed generations.';
        }
    },

    isModelServeReady(modelId) {
        if (!modelId) return true;
        const summary = App.state.registrySummary || {};
        const models = summary.models || [];
        const model = models.find((m) => m.model_id === modelId);
        return model ? Boolean(model.can_serve) : false;
    },

    // ---- Render the registered nodes grid ----
    renderNodes(nodes) {
        if (!nodes || nodes.length === 0) {
            this.nodesList.innerHTML = '<p class="nodes-empty">No nodes registered</p>';
            return;
        }

        const modelFilter = this.modelFilterEl?.value || '';
        const typeFilter = this.typeFilterEl?.value || '';
        const healthyOnly = Boolean(this.healthyOnlyEl?.checked);
        let filtered = nodes.slice();

        if (modelFilter) {
            filtered = filtered.filter((node) => {
                if (!node || !node.model_id) {
                    return node?.node_type === 'daemon';
                }
                return node.model_id === modelFilter;
            });
        }

        if (typeFilter) {
            filtered = filtered.filter((node) => (node.node_type || '') === typeFilter);
        }

        if (healthyOnly) {
            filtered = filtered.filter((node) => {
                if (!node?.model_id) return true;
                return this.isModelServeReady(node.model_id);
            });
        }

        if (!filtered.length) {
            this.nodesList.innerHTML = '<p class="nodes-empty">No nodes match the current filters</p>';
            return;
        }

        this.nodesList.innerHTML = '';
        // Sort infra first, then execution nodes.
        const sorted = [...filtered].sort((a, b) => {
            const order = {
                daemon: 0,
                mpc: 1,
                compute: 2,
                vision: 3,
                guard: 4,
            };
            const ta = order[a.node_type] ?? 9;
            const tb = order[b.node_type] ?? 9;
            if (ta !== tb) return ta - tb;
            const sa = Number(a.shard_index ?? 9999);
            const sb = Number(b.shard_index ?? 9999);
            if (sa !== sb) return sa - sb;
            return String(a.address || "").localeCompare(String(b.address || ""));
        });

        sorted.forEach(node => {
            const card = document.createElement('div');
            card.className = `node-card type-${node.node_type}`;

            const typeClass = node.node_type || 'compute';
            const typeLabel = typeClass;
            const layerInfo = node.node_function
                || (node.node_type === 'guard'
                    ? 'Guard relay'
                    : `Layers ${node.layer_start}-${node.layer_end - 1}`);

            const flags = [];
            if (node.has_embedding) flags.push('embed');
            if (node.has_lm_head) flags.push('lm_head');

            card.innerHTML = `
                <div class="node-card-header">
                    <span class="node-card-type ${typeClass}">${typeLabel}</span>
                    <span class="node-card-shard">${
                        (node.node_type === 'compute' || node.node_type === 'mpc' || node.node_type === 'vision')
                            ? ('Shard ' + node.shard_index)
                            : ''
                    }</span>
                </div>
                <div class="node-card-addr">${node.address}</div>
                <div class="node-card-layers">${layerInfo}${flags.length ? ' | ' + flags.join(', ') : ''}</div>
            `;
            this.nodesList.appendChild(card);
        });
    },

    // ---- Render the Tor-style circuit diagram ----
    renderCircuit(circuitMsg) {
        this.circuitHint.style.display = 'none';
        this.circuitContainer.style.display = 'block';
        this.circuitNodeEls = {};
        this.circuitArrowEls = {};

        // Vision pipeline
        if (circuitMsg.vision_nodes && circuitMsg.vision_nodes.length > 0) {
            this.visionSection.style.display = 'block';
            this.visionCircuitPath.innerHTML = '';
            this.buildCircuitPath(
                this.visionCircuitPath,
                circuitMsg.vision_nodes,
                circuitMsg.guard,
                'vision',
                circuitMsg
            );
        } else {
            this.visionSection.style.display = 'none';
        }

        // Text pipeline
        this.textCircuitPath.innerHTML = '';
        if (circuitMsg.text_nodes && circuitMsg.text_nodes.length > 0) {
            this.buildCircuitPath(
                this.textCircuitPath,
                circuitMsg.text_nodes,
                circuitMsg.guard,
                'text',
                circuitMsg
            );
        }
    },

    buildCircuitPath(container, nodes, guardAddr, phase, circuitMsg) {
        // Client node
        const clientEl = this.makeNodeEl({
            label: 'Client (you)',
            icon: '&#9673;',
            address: 'local',
            detail: 'Your machine',
            cssClass: 'client-node',
        });
        container.appendChild(clientEl);

        let prevAddr = 'client';

        // Guard node (if present)
        if (guardAddr) {
            const arrowEl = this.makeArrowEl('encrypted', prevAddr, guardAddr);
            container.appendChild(arrowEl);

            const guardEl = this.makeNodeEl({
                label: 'Guard',
                icon: '&#9681;',
                address: guardAddr,
                detail: 'Hides your IP',
                cssClass: 'guard',
            });
            container.appendChild(guardEl);
            this.circuitNodeEls[guardAddr] = guardEl;
            prevAddr = guardAddr;
        }

        // Shard nodes
        nodes.forEach((node, i) => {
            const onionLayer = nodes.length - i;
            const arrowLabel = i === nodes.length - 1 ? 'clear' : `onion ${onionLayer}`;
            const arrowEl = this.makeArrowEl(arrowLabel, prevAddr, node.address);
            container.appendChild(arrowEl);

            const isVision = phase === 'vision';
            const isMPC = node.node_type === 'mpc';
            let label, detail, icon, cssClass;

            if (isVision) {
                label = i === 0 ? 'MPC Vision' : `Vision ${node.shard_index}`;
                detail = i === 0 ? 'Secret-shared Conv3D' : 'ViT blocks + Merger';
                icon = '&#9673;';
                cssClass = 'vision-node';
                const nodeEl = this.makeNodeEl({ label, icon, address: node.address, detail, cssClass });
                container.appendChild(nodeEl);
                this.circuitNodeEls[node.address] = nodeEl;
            } else if (isMPC) {
                // Render MPC as a pair of two nodes (A + B) side by side
                const mpcPair = circuitMsg.mpc_pair || {};
                const pairEl = this.makeMPCPairEl(
                    mpcPair.node_a || node.address,
                    mpcPair.node_b || 'peer',
                    node.layer_start || 0,
                    node.layer_end || 6,
                );
                container.appendChild(pairEl);
                this.circuitNodeEls[node.address] = pairEl;
            } else {
                label = `Shard ${node.shard_index}`;
                detail = `Layers ${node.layer_start}-${node.layer_end - 1}`;
                icon = '&#9670;';
                cssClass = '';
                const nodeEl = this.makeNodeEl({ label, icon, address: node.address, detail, cssClass });
                container.appendChild(nodeEl);
                this.circuitNodeEls[node.address] = nodeEl;
            }

            prevAddr = node.address;
        });
    },

    makeMPCPairEl(addrA, addrB, layerStart, layerEnd) {
        const el = document.createElement('div');
        el.className = 'circuit-node mpc-pair-node';
        el.innerHTML = `
            <div class="mpc-pair-header">
                <div class="node-icon">&#9670;</div>
                <div class="node-label">MPC Shard 0</div>
            </div>
            <div class="mpc-pair-body">
                <div class="mpc-half mpc-node-a">
                    <div class="mpc-role">Node A (entry)</div>
                    <div class="mpc-addr">${addrA}</div>
                    <div class="mpc-task">Embedding + MPC layer 0</div>
                    <div class="mpc-task">Cleartext layers 1-${layerEnd - 1}</div>
                </div>
                <div class="mpc-link">
                    <svg width="24" height="30" viewBox="0 0 24 30">
                        <line x1="2" y1="15" x2="22" y2="15" stroke="var(--green)" stroke-width="2" stroke-dasharray="3 2"/>
                        <line x1="22" y1="15" x2="2" y2="15" stroke="var(--green)" stroke-width="2"/>
                    </svg>
                    <div class="mpc-link-label">shares</div>
                </div>
                <div class="mpc-half mpc-node-b">
                    <div class="mpc-role">Node B (peer)</div>
                    <div class="mpc-addr">${addrB}</div>
                    <div class="mpc-task">Holds share B</div>
                    <div class="mpc-task">MPC protocol co-compute</div>
                </div>
            </div>
            <div class="mpc-pair-footer">Secret-shared — neither node sees raw tokens</div>
        `;
        return el;
    },

    makeNodeEl({ label, icon, address, detail, cssClass }) {
        const el = document.createElement('div');
        el.className = `circuit-node ${cssClass || ''}`;
        el.innerHTML = `
            <div class="node-icon">${icon}</div>
            <div class="node-label">${label}</div>
            <div class="node-addr">${address}</div>
            <div class="node-detail">${detail || ''}</div>
        `;
        return el;
    },

    makeArrowEl(label, fromAddr, toAddr) {
        const key = `${fromAddr}->${toAddr}`;
        const el = document.createElement('div');
        el.className = 'circuit-arrow';
        el.innerHTML = `
            <div style="display:flex;flex-direction:column;align-items:center;gap:2px;">
                <div class="arrow-label">${label}</div>
                <svg width="40" height="16" viewBox="0 0 40 16">
                    <line x1="0" y1="8" x2="32" y2="8" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>
                    <polygon points="32,3 40,8 32,13" fill="currentColor"/>
                </svg>
            </div>
        `;
        this.circuitArrowEls[key] = el;
        return el;
    },

    // ---- Animate hops during inference ----
    animateHop(msg) {
        const nodeEl = this.circuitNodeEls[msg.address];
        if (nodeEl) {
            // Mark previous nodes as completed
            Object.values(this.circuitNodeEls).forEach(el => {
                if (el.classList.contains('active')) {
                    el.classList.remove('active');
                    el.classList.add('completed');
                }
            });

            nodeEl.classList.add('active');

            // Add latency display
            if (msg.latency_ms) {
                let latencyEl = nodeEl.querySelector('.node-latency');
                if (!latencyEl) {
                    latencyEl = document.createElement('div');
                    latencyEl.className = 'node-latency';
                    nodeEl.appendChild(latencyEl);
                }
                latencyEl.textContent = `${msg.latency_ms}ms`;
            }
        }

        // Animate arrows
        Object.entries(this.circuitArrowEls).forEach(([key, el]) => {
            if (key.endsWith('->' + msg.address)) {
                el.classList.add('completed');
            }
        });
    },
};

document.addEventListener('DOMContentLoaded', () => Network.init());
