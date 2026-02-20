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

        this.refreshBtn.addEventListener('click', () => this.loadNodes());

        // Listen for circuit updates from chat
        App.on('circuitUpdate', (msg) => this.renderCircuit(msg));
        App.on('hopUpdate', (msg) => this.animateHop(msg));
        App.on('nodesLoaded', (nodes) => this.renderNodes(nodes));
        App.on('tabSwitch', (tab) => {
            if (tab === 'network') this.loadNodes();
        });

        // Initial load
        this.loadNodes();
    },

    async loadNodes() {
        const data = await App.fetchJson('/api/network/nodes');
        if (data && data.nodes) {
            App.state.nodes = data.nodes;
            this.renderNodes(data.nodes);
        }
    },

    // ---- Render the registered nodes grid ----
    renderNodes(nodes) {
        if (!nodes || nodes.length === 0) {
            this.nodesList.innerHTML = '<p class="nodes-empty">No nodes registered</p>';
            return;
        }

        this.nodesList.innerHTML = '';
        // Sort: guard first, then vision, then compute by shard
        const sorted = [...nodes].sort((a, b) => {
            const order = { guard: 0, vision: 1, compute: 2 };
            const ta = order[a.node_type] ?? 2;
            const tb = order[b.node_type] ?? 2;
            if (ta !== tb) return ta - tb;
            return a.shard_index - b.shard_index;
        });

        sorted.forEach(node => {
            const card = document.createElement('div');
            card.className = `node-card type-${node.node_type}`;

            const typeClass = node.node_type || 'compute';
            const layerInfo = node.node_type === 'guard'
                ? 'Guard relay'
                : `Layers ${node.layer_start}-${node.layer_end - 1}`;

            const flags = [];
            if (node.has_embedding) flags.push('embed');
            if (node.has_lm_head) flags.push('lm_head');

            card.innerHTML = `
                <div class="node-card-header">
                    <span class="node-card-type ${typeClass}">${typeClass}</span>
                    <span class="node-card-shard">${node.node_type !== 'guard' ? 'Shard ' + node.shard_index : ''}</span>
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
