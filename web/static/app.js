/**
 * CoVayAI Web Arena — Interactive Client Logic
 * Handles Canvas rendering, Web Audio effects, MCTS state sync, and game loop.
 */

// Coordinate letters in standard Go (skips 'I')
const COL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'];

class GoWebArena {
    constructor() {
        this.boardSize = 9;
        this.grid = [];
        this.currentPlayer = 1; // 1 = Black, 2 = White
        this.userColor = 1;
        this.moveHistory = [];
        this.legalMoves = [];
        this.isGameOver = false;
        this.isAiThinking = false;
        this.soundEnabled = true;

        // Hover coordinate
        this.hoverCol = -1;
        this.hoverRow = -1;

        // Canvas & rendering
        this.canvas = document.getElementById('goCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.boardContainer = document.getElementById('boardContainer');

        // Audio Context (initialized on first click)
        this.audioCtx = null;

        // Cache DOM elements
        this.dom = {
            deviceText: document.getElementById('deviceText'),
            soundBtn: document.getElementById('soundBtn'),
            lastMoveText: document.getElementById('lastMoveText'),
            statusHint: document.getElementById('statusHint'),
            playerBlackCard: document.getElementById('playerBlackCard'),
            playerWhiteCard: document.getElementById('playerWhiteCard'),
            blackRole: document.getElementById('blackRole'),
            whiteRole: document.getElementById('whiteRole'),
            blackCaptures: document.getElementById('blackCaptures'),
            whiteCaptures: document.getElementById('whiteCaptures'),
            komiTag: document.getElementById('komiTag'),
            evalScore: document.getElementById('evalScore'),
            winrateBlackBar: document.getElementById('winrateBlackBar'),
            winrateWhiteBar: document.getElementById('winrateWhiteBar'),
            labelBlackWR: document.getElementById('labelBlackWR'),
            labelWhiteWR: document.getElementById('labelWhiteWR'),
            candidatesChips: document.getElementById('candidatesChips'),
            passBtn: document.getElementById('passBtn'),
            resignBtn: document.getElementById('resignBtn'),
            newGameBtn: document.getElementById('newGameBtn'),
            boardSizeSelect: document.getElementById('boardSizeSelect'),
            userColorSelect: document.getElementById('userColorSelect'),
            simsSelect: document.getElementById('simsSelect'),
            moveCounter: document.getElementById('moveCounter'),
            historyList: document.getElementById('historyList'),
            modalBackdrop: document.getElementById('modalBackdrop'),
            modalTitle: document.getElementById('modalTitle'),
            modalMessage: document.getElementById('modalMessage'),
            modalBlackScore: document.getElementById('modalBlackScore'),
            modalWhiteScore: document.getElementById('modalWhiteScore'),
            modalCloseBtn: document.getElementById('modalCloseBtn'),
        };

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Fetch system info and initial state
        await this.fetchServerInfo();
        await this.fetchState();
    }

    /* ==========================================================================
       Canvas Setup & Coordinate Math
       ========================================================================== */

    resizeCanvas() {
        // Container size determination (responsive max ~540px)
        const containerWidth = Math.min(window.innerWidth - 64, 520);
        const size = Math.max(300, containerWidth);

        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = size * dpr;
        this.canvas.height = size * dpr;
        this.canvas.style.width = `${size}px`;
        this.canvas.style.height = `${size}px`;

        this.ctx.scale(dpr, dpr);
        this.displaySize = size;
        this.margin = Math.round(size * 0.115);
        this.cellSize = (size - 2 * this.margin) / (this.boardSize - 1);
        this.stoneRadius = (this.cellSize / 2) * 0.94;

        this.render();
    }

    getIntersectionPos(col, row) {
        return {
            x: this.margin + col * this.cellSize,
            y: this.margin + row * this.cellSize,
        };
    }

    canvasCoordsToGrid(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        const col = Math.round((x - this.margin) / this.cellSize);
        const row = Math.round((y - this.margin) / this.cellSize);

        if (col >= 0 && col < this.boardSize && row >= 0 && row < this.boardSize) {
            // Check distance from actual intersection
            const pos = this.getIntersectionPos(col, row);
            const dist = Math.hypot(x - pos.x, y - pos.y);
            if (dist <= this.cellSize * 0.48) {
                return { col, row };
            }
        }
        return { col: -1, row: -1 };
    }

    actionToCoord(action) {
        if (action === this.boardSize * this.boardSize) {
            return "Pass";
        }
        const row = Math.floor(action / this.boardSize);
        const col = action % this.boardSize;
        const letter = COL_LETTERS[col] || '?';
        const num = this.boardSize - row;
        return `${letter}${num}`;
    }

    /* ==========================================================================
       Rendering: Board, Grid, Star points, Stones, Highlights
       ========================================================================== */

    render() {
        const ctx = this.ctx;
        const size = this.displaySize;
        ctx.clearRect(0, 0, size, size);

        this.drawCoordinates();
        this.drawGridLines();
        this.drawStarPoints();
        this.drawStones();
        this.drawGhostPreview();
        this.drawLastMoveMarker();
    }

    drawCoordinates() {
        const ctx = this.ctx;
        ctx.fillStyle = '#4a250a';
        ctx.font = `700 ${Math.max(11, Math.round(this.margin * 0.28))}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const textOffset = this.margin * 0.38;

        // Column letters (Top and Bottom)
        for (let col = 0; col < this.boardSize; col++) {
            const letter = COL_LETTERS[col];
            const x = this.margin + col * this.cellSize;
            ctx.fillText(letter, x, textOffset);
            ctx.fillText(letter, x, this.displaySize - textOffset);
        }

        // Row numbers (Left and Right)
        for (let row = 0; row < this.boardSize; row++) {
            const num = (this.boardSize - row).toString();
            const y = this.margin + row * this.cellSize;
            ctx.fillText(num, textOffset, y);
            ctx.fillText(num, this.displaySize - textOffset, y);
        }
    }

    drawGridLines() {
        const ctx = this.ctx;
        ctx.strokeStyle = '#4a280d';
        ctx.lineWidth = 1.2;

        for (let i = 0; i < this.boardSize; i++) {
            // Horizontal lines
            const y = Math.round(this.margin + i * this.cellSize) + 0.5;
            ctx.beginPath();
            ctx.moveTo(this.margin, y);
            ctx.lineTo(this.displaySize - this.margin, y);
            ctx.stroke();

            // Vertical lines
            const x = Math.round(this.margin + i * this.cellSize) + 0.5;
            ctx.beginPath();
            ctx.moveTo(x, this.margin);
            ctx.lineTo(x, this.displaySize - this.margin);
            ctx.stroke();
        }
    }

    drawStarPoints() {
        const ctx = this.ctx;
        ctx.fillStyle = '#4a280d';
        const starRadius = Math.max(2.5, this.cellSize * 0.08);

        let points = [];
        if (this.boardSize === 9) {
            points = [
                { c: 2, r: 2 }, { c: 6, r: 2 },
                { c: 4, r: 4 },
                { c: 2, r: 6 }, { c: 6, r: 6 },
            ];
        } else if (this.boardSize === 5) {
            points = [{ c: 2, r: 2 }];
        }

        points.forEach(pt => {
            const pos = this.getIntersectionPos(pt.c, pt.r);
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, starRadius, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawStones() {
        if (!this.grid || this.grid.length === 0) return;

        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                const color = this.grid[r][c];
                if (color === 1 || color === 2) {
                    const pos = this.getIntersectionPos(c, r);
                    this.drawStone(pos.x, pos.y, color);
                }
            }
        }
    }

    drawStone(x, y, color, opacity = 1.0) {
        const ctx = this.ctx;
        const r = this.stoneRadius;

        ctx.save();
        ctx.globalAlpha = opacity;

        // Realistic drop shadow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetX = 2.5;
        ctx.shadowOffsetY = 3.5;

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);

        // 3D Stone Radial Gradient
        const grad = ctx.createRadialGradient(
            x - r * 0.32, y - r * 0.32, r * 0.08,
            x, y, r
        );

        if (color === 1) {
            // Black Stone: Obsidian slate
            grad.addColorStop(0, '#52627a');
            grad.addColorStop(0.35, '#1e293b');
            grad.addColorStop(0.85, '#090d16');
            grad.addColorStop(1, '#020408');
        } else {
            // White Stone: Clam shell pearl
            grad.addColorStop(0, '#ffffff');
            grad.addColorStop(0.4, '#f1f5f9');
            grad.addColorStop(0.85, '#cbd5e1');
            grad.addColorStop(1, '#94a3b8');
        }

        ctx.fillStyle = grad;
        ctx.fill();

        // Subtle outer rim highlight for white stones
        if (color === 2) {
            ctx.shadowColor = 'transparent';
            ctx.lineWidth = 0.8;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
            ctx.stroke();
        }

        ctx.restore();
    }

    drawGhostPreview() {
        if (this.isGameOver || this.isAiThinking) return;
        if (this.currentPlayer !== this.userColor) return;
        if (this.hoverCol < 0 || this.hoverRow < 0) return;

        // Check if intersection is empty and move is legal
        if (this.grid[this.hoverRow][this.hoverCol] !== 0) return;

        const action = this.hoverRow * this.boardSize + this.hoverCol;
        if (!this.legalMoves.includes(action)) return;

        const pos = this.getIntersectionPos(this.hoverCol, this.hoverRow);
        this.drawStone(pos.x, pos.y, this.userColor, 0.45);

        // Subtle guide circle
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, this.stoneRadius + 2, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    drawLastMoveMarker() {
        if (!this.moveHistory || this.moveHistory.length === 0) return;

        const lastAction = this.moveHistory[this.moveHistory.length - 1];
        if (lastAction === this.boardSize * this.boardSize) return; // Pass move

        const row = Math.floor(lastAction / this.boardSize);
        const col = lastAction % this.boardSize;
        const pos = this.getIntersectionPos(col, row);

        const stoneColor = this.grid[row][col];
        const ctx = this.ctx;

        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, this.stoneRadius * 0.36, 0, Math.PI * 2);

        if (stoneColor === 1) {
            // White ring on black stone
            ctx.strokeStyle = '#06b6d4';
            ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
        } else {
            // Dark ring on white stone
            ctx.strokeStyle = '#0891b2';
            ctx.fillStyle = 'rgba(8, 145, 178, 0.25)';
        }

        ctx.lineWidth = 2.2;
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    /* ==========================================================================
       Sound Effects (Web Audio API Synthesizer)
       ========================================================================== */

    playStoneSound() {
        if (!this.soundEnabled) return;
        try {
            if (!this.audioCtx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                this.audioCtx = new AudioContext();
            }
            if (this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }

            const ctx = this.audioCtx;
            const now = ctx.currentTime;

            // 1. Noise click (stone impact)
            const bufferSize = ctx.sampleRate * 0.04;
            const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
            const data = buffer.getChannelData(0);
            for (let i = 0; i < bufferSize; i++) {
                data[i] = Math.random() * 2 - 1;
            }

            const noise = ctx.createBufferSource();
            noise.buffer = buffer;

            const bandpass = ctx.createBiquadFilter();
            bandpass.type = 'bandpass';
            bandpass.frequency.setValueAtTime(1100, now);
            bandpass.Q.setValueAtTime(3.0, now);

            const noiseGain = ctx.createGain();
            noiseGain.gain.setValueAtTime(0.7, now);
            noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.035);

            noise.connect(bandpass);
            bandpass.connect(noiseGain);
            noiseGain.connect(ctx.destination);
            noise.start(now);

            // 2. Body resonance (wood table thud)
            const osc = ctx.createOscillator();
            const oscGain = ctx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(320, now);
            osc.frequency.exponentialRampToValueAtTime(140, now + 0.06);

            oscGain.gain.setValueAtTime(0.5, now);
            oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

            osc.connect(oscGain);
            oscGain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.07);
        } catch (e) {
            // Audio not allowed or failed silently
        }
    }

    /* ==========================================================================
       Event Listeners & User Interaction
       ========================================================================== */

    setupEventListeners() {
        // Sound toggle
        this.dom.soundBtn.addEventListener('click', () => {
            this.soundEnabled = !this.soundEnabled;
            this.dom.soundBtn.textContent = this.soundEnabled ? '🔊' : '🔇';
            this.dom.soundBtn.title = this.soundEnabled ? 'Tắt âm thanh' : 'Bật âm thanh';
        });

        // Canvas hover preview
        this.canvas.addEventListener('mousemove', (e) => {
            const { col, row } = this.canvasCoordsToGrid(e.clientX, e.clientY);
            if (col !== this.hoverCol || row !== this.hoverRow) {
                this.hoverCol = col;
                this.hoverRow = row;
                this.render();
            }
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hoverCol = -1;
            this.hoverRow = -1;
            this.render();
        });

        // Canvas click to move
        this.canvas.addEventListener('click', async (e) => {
            if (this.isGameOver || this.isAiThinking) return;
            if (this.currentPlayer !== this.userColor) return;

            const { col, row } = this.canvasCoordsToGrid(e.clientX, e.clientY);
            if (col >= 0 && row >= 0) {
                const action = row * this.boardSize + col;
                if (this.legalMoves.includes(action)) {
                    await this.playMove(action);
                }
            }
        });

        // Controls
        this.dom.passBtn.addEventListener('click', async () => {
            if (this.isGameOver || this.isAiThinking) return;
            if (this.currentPlayer !== this.userColor) return;
            await this.passTurn();
        });

        this.dom.resignBtn.addEventListener('click', async () => {
            if (this.isGameOver || this.isAiThinking) return;
            if (confirm("Bạn có chắc chắn muốn xin thua (Resign) ván cờ này?")) {
                await this.resignGame();
            }
        });

        this.dom.newGameBtn.addEventListener('click', async () => {
            await this.startNewGame();
        });

        this.dom.modalCloseBtn.addEventListener('click', async () => {
            this.dom.modalBackdrop.classList.remove('active');
            await this.startNewGame();
        });
    }

    /* ==========================================================================
       API Communication
       ========================================================================== */

    async fetchServerInfo() {
        try {
            const res = await fetch('/api/info');
            const data = await res.json();
            if (data.device_name) {
                this.dom.deviceText.textContent = `${data.device_name} (${data.device})`;
            }
        } catch (e) {
            this.dom.deviceText.textContent = "CoVayAI Local Server";
        }
    }

    async fetchState() {
        try {
            const res = await fetch('/api/state');
            const state = await res.json();
            this.applyState(state);
        } catch (e) {
            console.error("Failed to fetch state:", e);
        }
    }

    async playMove(action) {
        this.setThinking(true);
        this.playStoneSound();

        try {
            const res = await fetch('/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
            const state = await res.json();
            this.playStoneSound();
            this.applyState(state);
        } catch (e) {
            console.error("Move error:", e);
        } finally {
            this.setThinking(false);
        }
    }

    async passTurn() {
        this.setThinking(true);
        try {
            const res = await fetch('/api/pass', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const state = await res.json();
            this.applyState(state);
        } catch (e) {
            console.error("Pass error:", e);
        } finally {
            this.setThinking(false);
        }
    }

    async resignGame() {
        this.setThinking(true);
        try {
            const res = await fetch('/api/resign', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const state = await res.json();
            this.applyState(state);
            this.showGameOverModal("Bạn đã xin thua!", "AI giành chiến thắng.");
        } catch (e) {
            console.error("Resign error:", e);
        } finally {
            this.setThinking(false);
        }
    }

    async startNewGame() {
        const boardSize = parseInt(this.dom.boardSizeSelect.value, 10);
        const userColor = this.dom.userColorSelect.value;
        const sims = parseInt(this.dom.simsSelect.value, 10);

        this.setThinking(true);
        try {
            const res = await fetch('/api/new_game', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    board_size: boardSize,
                    user_color: userColor,
                    sims: sims,
                }),
            });
            const state = await res.json();
            this.applyState(state);
        } catch (e) {
            console.error("New game error:", e);
        } finally {
            this.setThinking(false);
        }
    }

    setThinking(thinking) {
        this.isAiThinking = thinking;
        if (thinking) {
            this.dom.statusHint.textContent = "CoVayAI đang tính toán (MCTS)...";
            this.dom.statusHint.style.color = "var(--accent-gold)";
            this.canvas.style.cursor = "wait";
            this.dom.passBtn.disabled = true;
            this.dom.resignBtn.disabled = true;
        } else {
            this.canvas.style.cursor = "pointer";
            this.dom.passBtn.disabled = false;
            this.dom.resignBtn.disabled = false;
        }
    }

    /* ==========================================================================
       State Synchronization & UI Updates
       ========================================================================== */

    applyState(state) {
        if (state.board_size !== this.boardSize) {
            this.boardSize = state.board_size;
            this.dom.boardSizeSelect.value = state.board_size.toString();
            this.resizeCanvas();
        }

        this.grid = state.grid;
        this.currentPlayer = state.current_player;
        this.userColor = state.user_color;
        this.moveHistory = state.move_history || [];
        this.legalMoves = state.legal_moves || [];
        this.isGameOver = state.is_game_over;

        // Synchronize roles
        if (this.userColor === 1) {
            this.dom.blackRole.textContent = "Bạn (Người chơi)";
            this.dom.whiteRole.textContent = "CoVayAI (ResNet)";
        } else {
            this.dom.blackRole.textContent = "CoVayAI (ResNet)";
            this.dom.whiteRole.textContent = "Bạn (Người chơi)";
        }

        // Komi tag
        this.dom.komiTag.textContent = `+${state.komi}`;

        // Turn indicator
        if (this.currentPlayer === 1) {
            this.dom.playerBlackCard.classList.add('active-turn');
            this.dom.playerWhiteCard.classList.remove('active-turn');
        } else {
            this.dom.playerWhiteCard.classList.add('active-turn');
            this.dom.playerBlackCard.classList.remove('active-turn');
        }

        // Status Hint
        if (this.isGameOver) {
            this.dom.statusHint.textContent = "Ván cờ đã kết thúc";
            this.dom.statusHint.style.color = "var(--text-dim)";
        } else if (this.currentPlayer === this.userColor) {
            const colorName = this.userColor === 1 ? "Đen" : "Trắng";
            this.dom.statusHint.textContent = `Đến lượt bạn đi quân (${colorName})`;
            this.dom.statusHint.style.color = "var(--accent-emerald)";
        } else {
            this.dom.statusHint.textContent = "CoVayAI đang suy nghĩ...";
            this.dom.statusHint.style.color = "var(--accent-gold)";
        }

        // Last move display
        if (this.moveHistory.length > 0) {
            const last = this.moveHistory[this.moveHistory.length - 1];
            const coord = this.actionToCoord(last);
            const player = (this.moveHistory.length % 2 === 1) ? "Đen" : "Trắng";
            this.dom.lastMoveText.textContent = `${coord} (${player} - nước ${this.moveHistory.length})`;
        } else {
            this.dom.lastMoveText.textContent = "Khai cuộc";
        }

        // Captures calculation
        this.updateCaptures();

        // Winrate gauge
        this.updateWinrate(state.ai_winrate, state.ai_value);

        // Top candidate chips
        this.updateCandidateMoves(state.top_moves);

        // Move history table
        this.updateMoveHistoryList();

        // Game over modal
        if (this.isGameOver) {
            let msg = "";
            if (state.winner === 1) msg = "Quân Đen chiến thắng!";
            else if (state.winner === 2) msg = "Quân Trắng chiến thắng!";
            else msg = "Ván cờ Hòa!";

            this.dom.modalBlackScore.textContent = `${state.black_score.toFixed(1)} mục`;
            this.dom.modalWhiteScore.textContent = `${state.white_score.toFixed(1)} mục`;
            this.showGameOverModal("Ván Đấu Kết Thúc", msg);
        }

        this.render();
    }

    updateCaptures() {
        let blackOnBoard = 0;
        let whiteOnBoard = 0;
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                if (this.grid[r][c] === 1) blackOnBoard++;
                if (this.grid[r][c] === 2) whiteOnBoard++;
            }
        }

        let totalBlackMoves = 0;
        let totalWhiteMoves = 0;
        this.moveHistory.forEach((act, idx) => {
            if (act !== this.boardSize * this.boardSize) {
                if (idx % 2 === 0) totalBlackMoves++;
                else totalWhiteMoves++;
            }
        });

        const blackCaptures = Math.max(0, totalWhiteMoves - whiteOnBoard);
        const whiteCaptures = Math.max(0, totalBlackMoves - blackOnBoard);

        this.dom.blackCaptures.textContent = blackCaptures;
        this.dom.whiteCaptures.textContent = whiteCaptures;
    }

    updateWinrate(aiWinrate, aiValue) {
        const wr = aiWinrate !== undefined ? aiWinrate : 50.0;
        const blackWR = this.userColor === 1 ? (100 - wr) : wr;
        const whiteWR = 100 - blackWR;

        this.dom.winrateBlackBar.style.width = `${blackWR.toFixed(1)}%`;
        this.dom.winrateWhiteBar.style.width = `${whiteWR.toFixed(1)}%`;

        this.dom.labelBlackWR.textContent = `Đen ${blackWR.toFixed(1)}%`;
        this.dom.labelWhiteWR.textContent = `Trắng ${whiteWR.toFixed(1)}%`;

        const scoreText = (aiValue >= 0 ? `+${aiValue}` : `${aiValue}`);
        this.dom.evalScore.textContent = `${wr.toFixed(1)}% (v=${scoreText})`;
    }

    updateCandidateMoves(topMoves) {
        this.dom.candidatesChips.innerHTML = "";
        if (!topMoves || topMoves.length === 0) {
            this.dom.candidatesChips.innerHTML = `<span class="chip">Đang tính toán...</span>`;
            return;
        }

        topMoves.forEach(m => {
            const chip = document.createElement('span');
            chip.className = 'chip';
            const coord = this.actionToCoord(m.action);
            chip.textContent = `${coord}: ${(m.prob * 100).toFixed(0)}%`;
            this.dom.candidatesChips.appendChild(chip);
        });
    }

    updateMoveHistoryList() {
        this.dom.moveCounter.textContent = `${this.moveHistory.length} nước`;

        if (this.moveHistory.length === 0) {
            this.dom.historyList.innerHTML = `<div class="history-empty">Chưa có nước cờ nào.</div>`;
            return;
        }

        this.dom.historyList.innerHTML = "";
        this.moveHistory.forEach((act, idx) => {
            const item = document.createElement('span');
            const isBlack = (idx % 2 === 0);
            item.className = `history-item ${isBlack ? 'black-move' : 'white-move'}`;
            const coord = this.actionToCoord(act);
            item.textContent = `${idx + 1}. ${coord}`;
            this.dom.historyList.appendChild(item);
        });

        // Auto-scroll to latest move
        this.dom.historyList.scrollTop = this.dom.historyList.scrollHeight;
    }

    showGameOverModal(title, message) {
        this.dom.modalTitle.textContent = title;
        this.dom.modalMessage.textContent = message;
        this.dom.modalBackdrop.classList.add('active');
    }
}

// Instantiate arena when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.arena = new GoWebArena();
});
