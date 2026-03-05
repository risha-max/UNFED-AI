/* ================================================================
   UNFED AI Dashboard — Chat Interface
   ================================================================ */

const Chat = {
    messagesEl: null,
    inputEl: null,
    sendBtn: null,
    imageInput: null,
    imagePreview: null,
    imagePreviewImg: null,
    imageRemoveBtn: null,

    pendingImage: null,       // { file, path, dataUrl }
    currentAssistantEl: null, // Currently streaming message element
    currentAssistantText: "",
    welcomeShown: true,
    history: [],
    maxContextTurns: 8,
    maxContextChars: 12000,
    maxStoredTurns: 60,
    historyStorageKey: "unfed_chat_history_v1",

    init() {
        this.messagesEl = document.getElementById('chatMessages');
        this.inputEl = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.imagePreviewImg = document.getElementById('imagePreviewImg');
        this.imageRemoveBtn = document.getElementById('imageRemoveBtn');
        this.loadHistory();

        // Send on Enter (Shift+Enter for newline)
        this.inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.send();
            }
        });

        // Auto-resize textarea
        this.inputEl.addEventListener('input', () => {
            this.inputEl.style.height = 'auto';
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 150) + 'px';
        });

        this.sendBtn.addEventListener('click', () => this.send());

        // Image upload
        this.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageFile(e.target.files[0]);
            }
        });

        this.imageRemoveBtn.addEventListener('click', () => this.clearImage());

        // Drag & drop on the chat area
        this.messagesEl.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        this.messagesEl.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.handleImageFile(files[0]);
            }
        });

        // WebSocket messages
        App.on('chatMessage', (msg) => this.onMessage(msg));
    },

    handleImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.pendingImage = { file, dataUrl: e.target.result, path: null };
            this.imagePreviewImg.src = e.target.result;
            this.imagePreview.style.display = 'inline-block';

            // Auto-select a discovered vision-capable model.
            const modelSel = document.getElementById('modelSelect');
            const visionOpt = modelSel.querySelector('option[data-vision="true"]:not([disabled])');
            if (visionOpt) modelSel.value = visionOpt.value;
        };
        reader.readAsDataURL(file);
    },

    clearImage() {
        this.pendingImage = null;
        this.imagePreview.style.display = 'none';
        this.imagePreviewImg.src = '';
        this.imageInput.value = '';
    },

    async send() {
        const prompt = this.inputEl.value.trim();
        if (!prompt && !this.pendingImage) return;
        if (App.state.generating) return;

        // Remove welcome
        if (this.welcomeShown) {
            const welcome = this.messagesEl.querySelector('.chat-welcome');
            if (welcome) welcome.remove();
            this.welcomeShown = false;
        }

        // Build user message
        const userMsg = this.addMessage('user', prompt, this.pendingImage?.dataUrl);

        // Gather settings
        const selectedModel = App.getSelectedModel();
        if (!selectedModel || !selectedModel.model_id) {
            this.addSystemMessage('No model selected. Choose an available model first.');
            return;
        }
        if (!selectedModel.can_serve) {
            this.addSystemMessage('Selected model is currently unavailable (incomplete shard coverage).');
            return;
        }
        const modelType = selectedModel.model_type;
        const useVoting = document.getElementById('useVoting').checked;
        const maxTokensEl = document.getElementById('maxTokens');
        const parsedMaxTokens = Number.parseInt(maxTokensEl?.value ?? '', 10);
        const maxTokens = Number.isFinite(parsedMaxTokens) && parsedMaxTokens > 0
            ? parsedMaxTokens
            : 100;

        // Upload image if present
        let imagePath = null;
        if (this.pendingImage) {
            App.setStatus('connected', 'Uploading image...');
            const result = await App.postFile('/api/upload-image', this.pendingImage.file);
            if (result && result.path) {
                imagePath = result.path;
            } else {
                this.addSystemMessage('Failed to upload image');
                return;
            }
            this.clearImage();
        }

        // Clear input
        this.inputEl.value = '';
        this.inputEl.style.height = 'auto';

        // Create assistant message placeholder
        this.currentAssistantEl = this.addMessage('assistant', '', null, true);
        this.currentAssistantText = "";

        // Send via WebSocket
        const composedPrompt = this.buildPromptWithContext(prompt);
        const payload = {
            prompt: composedPrompt,
            model_id: selectedModel.model_id,
            model_type: modelType,
            use_voting: useVoting,
            max_tokens: maxTokens,
            he_output: Boolean(App.state.heOutputEnabled),
        };
        if (imagePath) payload.image_path = imagePath;
        const sent = App.sendChatMessage(payload);
        if (!sent) {
            if (App.state.devAuthBypass) {
                this.addSystemMessage('Not connected. Enter wallet address in Settings, then try again.');
            } else {
                this.addSystemMessage('Not connected to server');
            }
            App.state.generating = false;
            this.currentAssistantText = "";
            return;
        }

        if (prompt) {
            this.recordTurn("user", prompt);
        } else if (imagePath) {
            this.recordTurn("user", "[Image uploaded]");
        }
        this.sendBtn.disabled = true;
    },

    onMessage(msg) {
        switch (msg.type) {
            case 'status':
                App.setStatus('connected', msg.message);
                break;

            case 'circuit':
                // Forward to network visualizer
                App.state.circuit = msg;
                App.emit('circuitUpdate', msg);
                break;

            case 'hop':
                App.emit('hopUpdate', msg);
                break;

            case 'token':
                if (this.currentAssistantEl) {
                    const textEl = this.currentAssistantEl.querySelector('.message-text');
                    // Remove cursor, add text, re-add cursor
                    const cursor = textEl.querySelector('.cursor');
                    if (cursor) cursor.remove();
                    textEl.textContent += msg.text;
                    this.currentAssistantText += msg.text;
                    textEl.appendChild(this.makeCursor());
                    this.scrollToBottom();
                }
                break;

            case 'done':
                if (this.currentAssistantEl) {
                    const textEl = this.currentAssistantEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.cursor');
                    if (cursor) cursor.remove();

                    // Add meta
                    const meta = document.createElement('div');
                    meta.className = 'message-meta';
                    const parts = [
                        `${msg.total_tokens} tokens`,
                        `${msg.total_time}s`,
                        `${msg.tokens_per_sec} tok/s`,
                    ];
                    if (msg.cost !== undefined) parts.push(`cost ${msg.cost} UNFED`);
                    if (msg.remaining_balance !== undefined) parts.push(`balance ${msg.remaining_balance} UNFED`);
                    meta.textContent = parts.join(' | ');
                    this.currentAssistantEl.querySelector('.message-body').appendChild(meta);
                }

                this.currentAssistantEl = null;
                App.state.generating = false;
                this.sendBtn.disabled = false;
                if (this.currentAssistantText.trim()) {
                    this.recordTurn("assistant", this.currentAssistantText);
                }
                this.currentAssistantText = "";

                // Update stats
                document.getElementById('statTokens').textContent = msg.total_tokens;
                document.getElementById('statTime').textContent = msg.total_time + 's';
                document.getElementById('statSpeed').textContent = msg.tokens_per_sec + ' tok/s';
                const statCost = document.getElementById('statCost');
                if (statCost) statCost.textContent = (msg.cost !== undefined) ? `${msg.cost} UNFED` : '—';
                const statBalance = document.getElementById('statBalance');
                if (statBalance) statBalance.textContent = (msg.remaining_balance !== undefined) ? `${msg.remaining_balance} UNFED` : '—';

                App.setStatus('connected', 'Ready');
                break;

            case 'error':
                this.addSystemMessage('Error: ' + msg.message);
                if (this.currentAssistantEl) {
                    const textEl = this.currentAssistantEl.querySelector('.message-text');
                    const cursor = textEl.querySelector('.cursor');
                    if (cursor) cursor.remove();
                }
                this.currentAssistantEl = null;
                App.state.generating = false;
                this.sendBtn.disabled = false;
                this.currentAssistantText = "";
                App.setStatus('error', 'Error');
                break;
        }
    },

    buildPromptWithContext(currentPrompt) {
        if (!currentPrompt) return currentPrompt;
        const recent = this.history.slice(-this.maxContextTurns);
        if (!recent.length) return currentPrompt;

        const lines = [];
        let budgetUsed = 0;
        for (let i = recent.length - 1; i >= 0; i -= 1) {
            const turn = recent[i];
            if (!turn || !turn.role || !turn.content) continue;
            const label = turn.role === "assistant" ? "Assistant" : "User";
            const content = String(turn.content).trim().replace(/\s+/g, " ");
            if (!content) continue;
            const line = `${label}: ${content}`;
            if (budgetUsed + line.length > this.maxContextChars && lines.length) {
                break;
            }
            lines.unshift(line);
            budgetUsed += line.length + 1;
        }

        return [
            "Continue this conversation using the provided context.",
            "Conversation context:",
            ...lines,
            `User: ${currentPrompt}`,
            "Assistant:",
        ].join("\n");
    },

    recordTurn(role, content) {
        const normalizedRole = role === "assistant" ? "assistant" : "user";
        const normalizedContent = String(content || "").trim();
        if (!normalizedContent) return;
        this.history.push({
            role: normalizedRole,
            content: normalizedContent,
            ts: Date.now(),
        });
        if (this.history.length > this.maxStoredTurns) {
            this.history = this.history.slice(-this.maxStoredTurns);
        }
        this.saveHistory();
    },

    loadHistory() {
        try {
            const raw = localStorage.getItem(this.historyStorageKey);
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (!Array.isArray(parsed)) return;
            this.history = parsed
                .filter((x) => x && (x.role === "user" || x.role === "assistant"))
                .map((x) => ({
                    role: x.role,
                    content: String(x.content || "").trim(),
                    ts: Number(x.ts || 0),
                }))
                .filter((x) => x.content);
            if (this.history.length > this.maxStoredTurns) {
                this.history = this.history.slice(-this.maxStoredTurns);
            }
        } catch (e) {
            this.history = [];
        }
    },

    saveHistory() {
        try {
            localStorage.setItem(this.historyStorageKey, JSON.stringify(this.history));
        } catch (e) {
            // Best effort only; chat should continue without persistence.
        }
    },

    addMessage(role, text, imageDataUrl = null, streaming = false) {
        const msg = document.createElement('div');
        msg.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';

        const body = document.createElement('div');
        body.className = 'message-body';

        if (imageDataUrl) {
            const img = document.createElement('img');
            img.className = 'message-image';
            img.src = imageDataUrl;
            body.appendChild(img);
        }

        const textEl = document.createElement('div');
        textEl.className = 'message-text';
        textEl.textContent = text;

        if (streaming) {
            textEl.appendChild(this.makeCursor());
        }

        body.appendChild(textEl);
        msg.appendChild(avatar);
        msg.appendChild(body);

        this.messagesEl.appendChild(msg);
        this.scrollToBottom();
        return msg;
    },

    addSystemMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message assistant';
        msg.innerHTML = `
            <div class="message-avatar" style="background:var(--red-dim);color:var(--red);">!</div>
            <div class="message-body" style="border-color:var(--red-dim);">
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>`;
        this.messagesEl.appendChild(msg);
        this.scrollToBottom();
    },

    makeCursor() {
        const span = document.createElement('span');
        span.className = 'cursor';
        return span;
    },

    scrollToBottom() {
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    },

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },
};

document.addEventListener('DOMContentLoaded', () => Chat.init());
