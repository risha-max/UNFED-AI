/* ================================================================
   UNFED AI Dashboard — Mini-Chain Block Explorer
   ================================================================ */

const Chain = {
    blocksList: null,
    refreshBtn: null,
    chainHeight: null,
    chainShares: null,
    chainSettlements: null,
    chainLatestHash: null,
    contributionsChart: null,
    settlementsList: null,

    init() {
        this.blocksList = document.getElementById('blocksList');
        this.refreshBtn = document.getElementById('refreshChainBtn');
        this.chainHeight = document.getElementById('chainHeight');
        this.chainShares = document.getElementById('chainShares');
        this.chainSettlements = document.getElementById('chainSettlements');
        this.chainLatestHash = document.getElementById('chainLatestHash');
        this.contributionsChart = document.getElementById('contributionsChart');
        this.settlementsList = document.getElementById('settlementsList');

        this.refreshBtn.addEventListener('click', () => this.loadAll());

        // Chain WebSocket updates
        App.on('chainMessage', (msg) => this.onChainMessage(msg));
        App.on('tabSwitch', (tab) => {
            if (tab === 'chain') this.loadAll();
        });

        // Initial load
        this.loadAll();
    },

    async loadAll() {
        await Promise.all([
            this.loadInfo(),
            this.loadBlocks(),
            this.loadNodeTotals(),
            this.loadSettlements(),
            this.loadFees(),
        ]);
    },

    async loadInfo() {
        const data = await App.fetchJson('/api/chain/info');
        if (data) {
            this.chainHeight.textContent = data.height ?? 0;
            this.chainShares.textContent = data.total_shares ?? 0;
            this.chainSettlements.textContent = data.settlements ?? 0;
            this.chainLatestHash.textContent = App.truncHash(data.latest_block_hash, 10);
        }
    },

    async loadBlocks() {
        const data = await App.fetchJson('/api/chain/blocks?start=0&limit=50');
        if (data && data.blocks) {
            this.renderBlocks(data.blocks);
        }
    },

    async loadNodeTotals() {
        const data = await App.fetchJson('/api/chain/node-totals');
        if (data && data.node_totals) {
            this.renderContributions(data.node_totals);
        }
    },

    async loadSettlements() {
        const data = await App.fetchJson('/api/chain/settlements');
        if (data && data.settlements) {
            this.renderSettlements(data.settlements);
        }
    },

    async loadFees() {
        const data = await App.fetchJson('/api/chain/fees');
        if (data) {
            this.renderFees(data);
        }
    },

    renderFees(data) {
        const baseFeeEl = document.getElementById('feeBaseFee');
        const utilPctEl = document.getElementById('feeUtilPct');
        const utilFillEl = document.getElementById('feeUtilFill');
        const estCostEl = document.getElementById('feeEstCost');
        const sugTipEl = document.getElementById('feeSugTip');

        if (!baseFeeEl) return;

        const baseFee = data.base_fee ?? 0;
        const util = data.utilization ?? 0;
        const estCost = data.estimated_cost_100 ?? 0;
        const sugTip = data.suggested_tip ?? 0;

        baseFeeEl.textContent = baseFee.toFixed(6);
        utilPctEl.textContent = (util * 100).toFixed(1) + '%';
        estCostEl.textContent = estCost.toFixed(6);
        sugTipEl.textContent = sugTip > 0 ? sugTip.toFixed(6) : 'none';

        const utilPct = Math.min(util * 100, 100);
        utilFillEl.style.width = utilPct + '%';

        // Color the utilization gauge based on load
        if (util > 0.9) {
            utilFillEl.className = 'utilization-fill util-high';
        } else if (util > 0.7) {
            utilFillEl.className = 'utilization-fill util-med';
        } else {
            utilFillEl.className = 'utilization-fill util-low';
        }

        // Show daemon status
        if (!data.daemon_available) {
            baseFeeEl.title = 'Using default fee (daemon not connected)';
        }
    },

    onChainMessage(msg) {
        if (msg.type === 'chain_info' && msg.info) {
            this.chainHeight.textContent = msg.info.height ?? 0;
            this.chainShares.textContent = msg.info.total_shares ?? 0;
            this.chainSettlements.textContent = msg.info.settlements ?? 0;
            this.chainLatestHash.textContent = App.truncHash(msg.info.latest_block_hash, 10);

            if (msg.info.node_totals) {
                this.renderContributions(msg.info.node_totals);
            }
        }

        if (msg.type === 'new_block' && msg.block) {
            this.prependBlock(msg.block);
        }
    },

    // ---- Blocks list ----
    renderBlocks(blocks) {
        if (!blocks || blocks.length === 0) {
            this.blocksList.innerHTML = '<p class="blocks-empty">No blocks yet. Send a query through the Chat tab to trigger inference &mdash; each shard computation produces a compute share, and shares are bundled into blocks.</p>';
            return;
        }

        this.blocksList.innerHTML = '';
        // Show newest first
        const reversed = [...blocks].reverse();
        reversed.forEach(block => {
            this.blocksList.appendChild(this.makeBlockCard(block));
        });
    },

    prependBlock(block) {
        const empty = this.blocksList.querySelector('.blocks-empty');
        if (empty) empty.remove();

        const card = this.makeBlockCard(block);
        card.style.animation = 'fadeIn 0.3s ease';
        this.blocksList.prepend(card);
    },

    makeBlockCard(block) {
        const card = document.createElement('div');
        card.className = 'block-card';
        if (block.index === 0) card.classList.add('genesis');

        const sharesCount = block.shares ? block.shares.length : 0;
        const timeStr = block.timestamp ? App.formatTime(block.timestamp) : '—';
        const isGenesis = block.index === 0;

        // Compute unique nodes and sessions in this block
        const uniqueNodes = new Set();
        const uniqueSessions = new Set();
        (block.shares || []).forEach(s => {
            if (s.node_id) uniqueNodes.add(s.node_id);
            if (s.session_id) uniqueSessions.add(s.session_id);
        });

        // Identify the proposer (the node whose shares are in this block)
        const proposer = uniqueNodes.size === 1 ? [...uniqueNodes][0] : null;
        const proposerLabel = proposer
            ? `<span class="block-proposer" title="${proposer}">&#9650; ${proposer.slice(0, 8)}...</span>`
            : (uniqueNodes.size > 1 ? `<span class="block-proposer">${uniqueNodes.size} nodes</span>` : '');

        // Identify which shard(s) contributed
        const shardSet = new Set();
        (block.shares || []).forEach(s => { if (s.shard_index !== undefined) shardSet.add(s.shard_index); });
        const shardLabel = shardSet.size > 0 ? `Shard${shardSet.size > 1 ? 's' : ''} ${[...shardSet].sort().join(', ')}` : '';

        const summaryParts = [];
        if (isGenesis) {
            summaryParts.push('Genesis block &mdash; chain anchor');
        } else {
            summaryParts.push(`${sharesCount} share${sharesCount !== 1 ? 's' : ''}`);
            if (shardLabel) summaryParts.push(shardLabel);
            if (uniqueSessions.size > 0) summaryParts.push(`${uniqueSessions.size} session${uniqueSessions.size !== 1 ? 's' : ''}`);
        }

        card.innerHTML = `
            <div class="block-card-header">
                <div class="block-card-title">
                    <span class="block-index">${isGenesis ? '&#9939; Genesis' : 'Block #' + block.index}</span>
                    ${proposerLabel}
                </div>
                <span class="block-time">${timeStr}</span>
            </div>
            <div class="block-card-summary">
                <span class="block-summary-text">${summaryParts.join(' &middot; ')}</span>
                <span class="block-hash">${App.truncHash(block.block_hash, 8)}</span>
            </div>
            <div class="block-details">
                ${isGenesis ? '<div class="block-genesis-note">The genesis block is the fixed starting point of the chain. All nodes start from this same block so they can validate each other\'s chains. It contains no shares (no computation happened yet).</div>' : ''}
                ${this.renderShares(block.shares || [])}
                <div class="block-meta">
                    <div class="block-meta-row">
                        <span class="block-meta-label">Block hash</span>
                        <span class="block-meta-value">${block.block_hash || '—'}</span>
                    </div>
                    <div class="block-meta-row">
                        <span class="block-meta-label">Previous hash</span>
                        <span class="block-meta-value">${block.previous_hash || '—'}</span>
                    </div>
                    <div class="block-meta-row">
                        <span class="block-meta-label">Timestamp</span>
                        <span class="block-meta-value">${block.timestamp ? new Date(block.timestamp * 1000).toISOString() : '—'}</span>
                    </div>
                </div>
            </div>
        `;

        card.addEventListener('click', () => {
            card.classList.toggle('expanded');
        });

        return card;
    },

    renderShares(shares) {
        if (shares.length === 0) {
            return '';
        }

        let html = '<div class="shares-header-row"><span>Node</span><span>Shard</span><span>Weight</span><span>Tokens</span><span>Hash</span></div>';
        shares.forEach(share => {
            const nodeShort = share.node_id ? share.node_id.slice(0, 8) : '—';
            const isMPC = share.node_id && share.node_id.startsWith('peer-of-');
            const isGuard = share.shard_index === -1;
            const shardLabel = isGuard
                ? 'Guard'
                : (share.shard_index === 0
                    ? (isMPC ? 'MPC-B' : 'Shard 0')
                    : `Shard ${share.shard_index}`);
            const nodeClass = isMPC ? 'share-node mpc-peer' : (isGuard ? 'share-node guard-node' : 'share-node');
            const weight = share.share_weight !== undefined ? share.share_weight : 1.0;
            const weightLabel = weight === 1.0 ? '1.0' : weight.toFixed(2);
            const weightClass = weight < 1.0 ? 'share-weight fractional' : 'share-weight';

            html += `
                <div class="share-row">
                    <span class="${nodeClass}" title="${share.node_id || ''}">${nodeShort}...</span>
                    <span class="share-shard">${shardLabel}</span>
                    <span class="${weightClass}">${weightLabel}</span>
                    <span class="share-tokens">${share.tokens_processed} tok</span>
                    <span class="share-hash" title="${share.activation_hash || ''}">${App.truncHash(share.activation_hash, 8)}</span>
                </div>
            `;
        });
        return html;
    },

    // ---- Contributions chart ----
    renderContributions(totals) {
        if (!totals || Object.keys(totals).length === 0) {
            this.contributionsChart.innerHTML = '<p class="chart-empty">No contributions yet</p>';
            return;
        }

        const maxVal = Math.max(...Object.values(totals), 1);
        this.contributionsChart.innerHTML = '';

        // Sort by contribution (desc)
        const sorted = Object.entries(totals).sort((a, b) => b[1] - a[1]);

        sorted.forEach(([nodeId, count]) => {
            const pct = (count / maxVal) * 100;
            const isGuard = count < 1 && count > 0; // guard shares are fractional
            const displayCount = Number.isInteger(count) ? count : count.toFixed(2);
            const roleTag = isGuard ? ' <span class="role-tag guard-tag">guard</span>' : '';
            const bar = document.createElement('div');
            bar.className = 'contribution-bar';
            bar.innerHTML = `
                <div class="contribution-bar-header">
                    <span class="contribution-bar-label">${nodeId.slice(0, 8)}...${roleTag}</span>
                    <span class="contribution-bar-value">${displayCount} weighted shares</span>
                </div>
                <div class="contribution-bar-track">
                    <div class="contribution-bar-fill" style="width:${pct}%"></div>
                </div>
            `;
            this.contributionsChart.appendChild(bar);
        });
    },

    // ---- Settlements ----
    renderSettlements(settlements) {
        if (!settlements || settlements.length === 0) {
            this.settlementsList.innerHTML = '<p class="settlements-empty">No settlements yet &mdash; need at least 6 blocks of compute activity.</p>';
            return;
        }

        this.settlementsList.innerHTML = '';
        // Show newest first
        const reversed = [...settlements].reverse();
        reversed.forEach((s, i) => {
            const card = document.createElement('div');
            card.className = 'settlement-card';

            const blockRange = s.block_range
                ? `Blocks ${s.block_range[0]}&ndash;${s.block_range[1]}`
                : '—';
            const nodeCount = s.node_shares ? Object.keys(s.node_shares).length : 0;

            const displayTotal = Number.isInteger(s.total_shares) ? s.total_shares : parseFloat(s.total_shares).toFixed(2);
            card.innerHTML = `
                <div class="settlement-card-header">
                    <span class="settlement-round">Settlement #${reversed.length - i}</span>
                    <span class="settlement-total">${displayTotal} weighted shares</span>
                </div>
                <div class="settlement-card-detail">
                    <span class="settlement-blocks">${blockRange}</span>
                    ${nodeCount > 0 ? `<span class="settlement-nodes">${nodeCount} node${nodeCount !== 1 ? 's' : ''} credited</span>` : ''}
                </div>
                ${s.node_shares ? this.renderSettlementBreakdown(s.node_shares, s.total_shares) : ''}
                <div class="settlement-hash">${App.truncHash(s.settlement_hash, 14)}</div>
            `;
            this.settlementsList.appendChild(card);
        });
    },

    renderSettlementBreakdown(nodeShares, totalShares) {
        if (!nodeShares || Object.keys(nodeShares).length === 0) return '';
        let html = '<div class="settlement-breakdown">';
        const sorted = Object.entries(nodeShares).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([nodeId, count]) => {
            const pct = totalShares > 0 ? ((count / totalShares) * 100).toFixed(1) : 0;
            const displayCount = Number.isInteger(count) ? count : parseFloat(count).toFixed(2);
            html += `
                <div class="settlement-row">
                    <span class="settlement-row-node" title="${nodeId}">${nodeId.slice(0, 8)}...</span>
                    <span class="settlement-row-bar"><span class="settlement-row-fill" style="width:${pct}%"></span></span>
                    <span class="settlement-row-pct">${pct}%</span>
                    <span class="settlement-row-count">${displayCount}</span>
                </div>
            `;
        });
        html += '</div>';
        return html;
    },
};

document.addEventListener('DOMContentLoaded', () => Chain.init());
