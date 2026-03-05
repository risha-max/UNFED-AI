/* ================================================================
   UNFED AI Dashboard — Mini-Chain Block Explorer
   ================================================================ */

const Chain = {
    blocksList: null,
    refreshBtn: null,
    searchInput: null,
    pageNewestBtn: null,
    pageNewerBtn: null,
    pageOlderBtn: null,
    pageOldestBtn: null,
    pageInfoEl: null,
    subnavEl: null,
    viewButtons: [],
    views: {},
    chainHeight: null,
    chainShares: null,
    chainSettlements: null,
    chainLatestHash: null,
    contributionsChart: null,
    settlementsList: null,
    infraPayoutSummary: null,
    allBlocks: [],
    pageSize: 50,
    currentPage: 0,  // 0 = newest page
    totalPages: 1,
    totalBlocks: 0,

    init() {
        this.blocksList = document.getElementById('blocksList');
        this.refreshBtn = document.getElementById('refreshChainBtn');
        this.searchInput = document.getElementById('chainSearch');
        this.pageNewestBtn = document.getElementById('chainPageNewest');
        this.pageNewerBtn = document.getElementById('chainPageNewer');
        this.pageOlderBtn = document.getElementById('chainPageOlder');
        this.pageOldestBtn = document.getElementById('chainPageOldest');
        this.pageInfoEl = document.getElementById('chainPageInfo');
        this.subnavEl = document.getElementById('chainSubnav');
        this.viewButtons = Array.from(document.querySelectorAll('[data-chain-view]'));
        this.views = {
            blocks: document.getElementById('chainViewBlocks'),
            fees: document.getElementById('chainViewFees'),
            contributions: document.getElementById('chainViewContributions'),
            settlements: document.getElementById('chainViewSettlements'),
        };
        this.chainHeight = document.getElementById('chainHeight');
        this.chainShares = document.getElementById('chainShares');
        this.chainSettlements = document.getElementById('chainSettlements');
        this.chainLatestHash = document.getElementById('chainLatestHash');
        this.contributionsChart = document.getElementById('contributionsChart');
        this.settlementsList = document.getElementById('settlementsList');
        this.infraPayoutSummary = document.getElementById('infraPayoutSummary');

        this.refreshBtn.addEventListener('click', () => this.loadAll());
        if (this.searchInput) {
            this.searchInput.addEventListener('input', () => this.renderBlocks(this.allBlocks));
        }
        this.viewButtons.forEach((btn) => {
            btn.addEventListener('click', () => this.setView(btn.dataset.chainView || 'blocks'));
        });
        if (this.pageNewestBtn) {
            this.pageNewestBtn.addEventListener('click', () => {
                this.currentPage = 0;
                this.loadBlocks();
            });
        }
        if (this.pageNewerBtn) {
            this.pageNewerBtn.addEventListener('click', () => {
                this.currentPage = Math.max(0, this.currentPage - 1);
                this.loadBlocks();
            });
        }
        if (this.pageOlderBtn) {
            this.pageOlderBtn.addEventListener('click', () => {
                this.currentPage = Math.min(this.totalPages - 1, this.currentPage + 1);
                this.loadBlocks();
            });
        }
        if (this.pageOldestBtn) {
            this.pageOldestBtn.addEventListener('click', () => {
                this.currentPage = Math.max(0, this.totalPages - 1);
                this.loadBlocks();
            });
        }

        // Chain WebSocket updates
        App.on('chainMessage', (msg) => this.onChainMessage(msg));
        App.on('tabSwitch', (tab) => {
            if (tab === 'chain') this.loadAll();
        });

        // Initial load
        this.setView('blocks');
        this.loadAll();
    },

    setView(view) {
        this.viewButtons.forEach((btn) => {
            btn.classList.toggle('active', (btn.dataset.chainView || '') === view);
        });
        Object.entries(this.views).forEach(([key, el]) => {
            if (!el) return;
            el.classList.toggle('active', key === view);
        });
    },

    async loadAll() {
        await Promise.all([
            this.loadInfo(),
            this.loadBlocks(),
            this.loadNodeTotals(),
            this.loadSettlements(),
            this.loadFees(),
            this.loadInfraPayouts(),
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
        // Paginate across the entire chain, newest page first.
        const info = await App.fetchJson('/api/chain/info');
        const height = Number(info?.height || 0);
        this.totalBlocks = Math.max(0, height + 1);
        this.totalPages = Math.max(1, Math.ceil(this.totalBlocks / this.pageSize));
        if (this.currentPage > this.totalPages - 1) {
            this.currentPage = this.totalPages - 1;
        }

        const endExclusive = Math.max(0, this.totalBlocks - (this.currentPage * this.pageSize));
        const start = Math.max(0, endExclusive - this.pageSize);
        const limit = Math.max(0, endExclusive - start);
        const data = await App.fetchJson(`/api/chain/blocks?start=${start}&limit=${limit}`);
        if (data && data.blocks) {
            // API returns oldest->newest, we display newest first.
            this.allBlocks = [...data.blocks].reverse();
            this.renderBlocks(this.allBlocks);
            this.updatePaginationUi();
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

    async loadInfraPayouts() {
        const data = await App.fetchJson('/api/network/health');
        if (data) {
            this.renderInfraPayouts(data);
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
            if (this.currentPage === 0) {
                this.prependBlock(msg.block);
            }
            // Keep pagination/height accurate on any page.
            this.loadInfo().then(() => this.updatePaginationUi());
        }
    },

    // ---- Blocks table ----
    renderBlocks(blocks) {
        const searchTerm = (this.searchInput?.value || '').trim().toLowerCase();
        const source = blocks || [];
        const filtered = source.filter((block) => this.matchesSearch(block, searchTerm));

        if (!filtered.length) {
            const emptyText = searchTerm
                ? 'No blocks match your search.'
                : 'No blocks yet. Send a query through the Chat tab to generate compute shares.';
            this.blocksList.innerHTML = `<tr><td colspan="7" class="blocks-empty">${emptyText}</td></tr>`;
            return;
        }

        this.blocksList.innerHTML = '';
        filtered.forEach((block) => {
            const { row, detailsRow } = this.makeBlockRows(block);
            this.blocksList.appendChild(row);
            this.blocksList.appendChild(detailsRow);
        });
    },

    prependBlock(block) {
        this.allBlocks = [block, ...this.allBlocks];
        if (this.allBlocks.length > this.pageSize) {
            this.allBlocks = this.allBlocks.slice(0, this.pageSize);
        }
        this.renderBlocks(this.allBlocks);
    },

    updatePaginationUi() {
        if (this.pageInfoEl) {
            this.pageInfoEl.textContent = `Page ${this.currentPage + 1} / ${this.totalPages}`;
        }
        if (this.pageNewestBtn) this.pageNewestBtn.disabled = this.currentPage === 0;
        if (this.pageNewerBtn) this.pageNewerBtn.disabled = this.currentPage === 0;
        if (this.pageOlderBtn) this.pageOlderBtn.disabled = this.currentPage >= this.totalPages - 1;
        if (this.pageOldestBtn) this.pageOldestBtn.disabled = this.currentPage >= this.totalPages - 1;
    },

    matchesSearch(block, searchTerm) {
        if (!searchTerm) return true;
        const flat = [
            String(block.index ?? ''),
            String(block.block_hash ?? ''),
            String(block.previous_hash ?? ''),
            ...(block.shares || []).map((s) => `${s.node_id || ''} ${s.session_id || ''} ${s.activation_hash || ''}`),
        ].join(' ').toLowerCase();
        return flat.includes(searchTerm);
    },

    formatAge(timestamp) {
        if (!timestamp) return '—';
        const sec = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
        if (sec < 60) return `${sec}s`;
        if (sec < 3600) return `${Math.floor(sec / 60)}m`;
        if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
        return `${Math.floor(sec / 86400)}d`;
    },

    makeBlockRows(block) {
        const row = document.createElement('tr');
        row.className = 'explorer-row';
        if (block.index === 0) row.classList.add('is-genesis');

        const detailsRow = document.createElement('tr');
        detailsRow.className = 'explorer-row-details';
        detailsRow.hidden = true;

        const sharesCount = block.shares ? block.shares.length : 0;
        const age = this.formatAge(block.timestamp);
        const isGenesis = block.index === 0;

        const uniqueNodes = new Set();
        const uniqueSessions = new Set();
        (block.shares || []).forEach(s => {
            if (s.node_id) uniqueNodes.add(s.node_id);
            if (s.session_id) uniqueSessions.add(s.session_id);
        });
        const proposer = uniqueNodes.size === 1 ? [...uniqueNodes][0] : null;

        const shardSet = new Set();
        (block.shares || []).forEach(s => { if (s.shard_index !== undefined) shardSet.add(s.shard_index); });
        const shardLabel = shardSet.size ? [...shardSet].sort().join(', ') : '—';
        const proposerLabel = proposer
            ? `${proposer.slice(0, 8)}...`
            : `${uniqueNodes.size || 0} nodes`;
        const fullTime = block.timestamp ? new Date(block.timestamp * 1000).toISOString() : '—';

        row.innerHTML = `
            <td class="col-block">${isGenesis ? '&#9939; Genesis' : `#${block.index}`}</td>
            <td title="${fullTime}">${age}</td>
            <td class="mono">${App.truncHash(block.block_hash, 10)}</td>
            <td class="mono" title="${proposer || ''}">${proposerLabel}</td>
            <td>${sharesCount}</td>
            <td>${uniqueSessions.size}</td>
            <td>${shardLabel}</td>
        `;

        detailsRow.innerHTML = `
            <td colspan="7">
                <div class="explorer-detail-wrap">
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
                            <span class="block-meta-value">${fullTime}</span>
                        </div>
                    </div>
                    ${this.renderShares(block.shares || [])}
                </div>
            </td>
        `;

        row.addEventListener('click', () => {
            const expanded = row.classList.toggle('expanded');
            detailsRow.hidden = !expanded;
        });

        return { row, detailsRow };
    },

    renderShares(shares) {
        if (shares.length === 0) {
            return '<div class="explorer-no-shares">No shares in this block.</div>';
        }

        let html = '<div class="shares-header-row"><span>Node</span><span>Shard</span><span>Weight</span><span>Tokens</span><span>Session</span><span>Hash</span></div>';
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
                    <span class="share-session" title="${share.session_id || ''}">${App.truncHash(share.session_id || '—', 6)}</span>
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
    renderInfraPayouts(data) {
        if (!this.infraPayoutSummary) return;

        const daemonMap = data.daemon_payout_share || {};
        const verifierMap = data.verifier_payout_share || {};
        const daemonWork = data.daemon_work_window || {};
        const verifierWork = data.verifier_work_window || {};
        const selectedDaemon = data.selected_daemon_recipient || '';
        const selectedVerifier = data.selected_verifier_recipient || '';

        const daemonRows = Object.entries(daemonMap).sort((a, b) => b[1] - a[1]);
        const verifierRows = Object.entries(verifierMap).sort((a, b) => b[1] - a[1]);
        const daemonWorkRows = Object.entries(daemonWork).sort((a, b) => b[1] - a[1]);
        const verifierWorkRows = Object.entries(verifierWork).sort((a, b) => b[1] - a[1]);

        if (
            !daemonRows.length &&
            !verifierRows.length &&
            !daemonWorkRows.length &&
            !verifierWorkRows.length &&
            !selectedDaemon &&
            !selectedVerifier
        ) {
            this.infraPayoutSummary.innerHTML = '<p class="settlements-empty">No daemon/verifier payout telemetry available yet.</p>';
            return;
        }

        const renderShareRows = (rows, selected) => rows.map(([addr, share]) => {
            const isSelected = selected && addr.toLowerCase() === selected.toLowerCase();
            return `
                <div class="infra-payout-row ${isSelected ? 'selected' : ''}">
                    <span class="infra-payout-address mono" title="${addr}">${App.truncHash(addr, 8)}</span>
                    <span class="infra-payout-share">${(Number(share) * 100).toFixed(1)}%</span>
                </div>
            `;
        }).join('');

        const renderWorkRows = (rows, selected) => rows.map(([addr, units]) => {
            const isSelected = selected && addr.toLowerCase() === selected.toLowerCase();
            return `
                <div class="infra-payout-row ${isSelected ? 'selected' : ''}">
                    <span class="infra-payout-address mono" title="${addr}">${App.truncHash(addr, 8)}</span>
                    <span class="infra-payout-work">${Number(units).toFixed(1)} units</span>
                </div>
            `;
        }).join('');

        this.infraPayoutSummary.innerHTML = `
            <div class="infra-payout-grid">
                <div class="infra-payout-card">
                    <div class="infra-payout-title">Daemon pending work</div>
                    ${daemonWorkRows.length ? renderWorkRows(daemonWorkRows, selectedDaemon) : '<div class="infra-empty">No daemon work yet</div>'}
                    <div class="infra-subtitle">Last payout share</div>
                    ${daemonRows.length ? renderShareRows(daemonRows, selectedDaemon) : '<div class="infra-empty">No daemon payout rows yet</div>'}
                </div>
                <div class="infra-payout-card">
                    <div class="infra-payout-title">Verifier pending work</div>
                    ${verifierWorkRows.length ? renderWorkRows(verifierWorkRows, selectedVerifier) : '<div class="infra-empty">No verifier work yet</div>'}
                    <div class="infra-subtitle">Last payout share</div>
                    ${verifierRows.length ? renderShareRows(verifierRows, selectedVerifier) : '<div class="infra-empty">No verifier payout rows yet</div>'}
                </div>
            </div>
            <div class="chain-card-hint">Pending work accumulates continuously; payout share updates after infra settlement accounting cycles.</div>
        `;
    },

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
