import type { StateSnapshot, RewardType } from '../types';
import { DragHandler } from '../interaction/dragHandler';
import { GameWSClient } from '../ws/client';
import { getLineColor } from '../renderer/lines';

const DAYS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];

const REWARD_LABELS: Record<number, string> = {
  0: 'Line',
  1: 'Locomotive',
  2: 'Tunnel',
  3: 'Carriage',
  4: 'Interchange',
};

export class HUD {
  private container: HTMLElement;
  private wsClient: GameWSClient;
  private dragHandler: DragHandler | null = null;

  private scoreText!: HTMLElement;
  private dayText!: HTMLElement;
  private clockHand!: SVGLineElement;

  private pauseBtn!: HTMLButtonElement;
  private playBtn!: HTMLButtonElement;
  private fastBtn!: HTMLButtonElement;

  private linesStack!: HTMLElement;
  private lineTokenBtn!: HTMLElement;
  private lineCount!: HTMLElement;
  private trainToken!: HTMLElement;
  private trainCount!: HTMLElement;
  private carriageToken!: HTMLElement;
  private carriageCount!: HTMLElement;
  private tunnelToken!: HTMLElement;
  private tunnelCount!: HTMLElement;
  private interchangeToken!: HTMLElement;
  private interchangeCount!: HTMLElement;

  private rewardModal!: HTMLElement;
  private rewardOptions!: HTMLElement;
  private gameOverModal!: HTMLElement;
  private toastEl!: HTMLElement;

  constructor(container: HTMLElement, wsClient: GameWSClient) {
    this.container = container;
    this.wsClient = wsClient;

    this.createDomElements();
    this.attachEventListeners();
  }

  public setDragHandler(handler: DragHandler): void {
    this.dragHandler = handler;
  }

  private createDomElements(): void {
    this.container.innerHTML = `
      <!-- Top Left Bar -->
      <div class="hud-top-left">
        <button id="hud-reset-btn" class="hud-icon-btn" title="Reset Game">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
        </button>
      </div>

      <!-- Top Right Bar (Day, Score, Analog Clock) -->
      <div class="hud-top-right">
        <span id="hud-score-text" class="hud-score">0</span>
        <span id="hud-day-text" class="hud-day">MON</span>
        <div class="hud-clock-container">
          <svg id="hud-clock-svg" width="34" height="34" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="15" fill="#e64b3c" stroke="#252525" stroke-width="2" />
            <line id="hud-clock-hand" x1="18" y1="18" x2="18" y2="6" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" />
          </svg>
        </div>
      </div>

      <!-- Far Right Control Dock (Speed & Line Tokens) -->
      <div class="hud-right-dock">
        <div class="hud-speed-controls">
          <button id="hud-pause-btn" class="hud-speed-btn" title="Pause">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          </button>
          <button id="hud-play-btn" class="hud-speed-btn active" title="Play">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          </button>
          <button id="hud-fast-btn" class="hud-speed-btn" title="Fast">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="3 3 13 12 3 21 3 3" />
              <polygon points="13 3 23 12 13 21 13 3" />
            </svg>
          </button>
        </div>

        <div id="hud-lines-stack" class="hud-lines-stack">
        </div>
      </div>

      <!-- Far Left Resource Dock (Line, Train, Carriage, Tunnel, Interchange) -->
      <div id="hud-left-dock" class="hud-left-dock">
        <div id="hud-line-token-btn" class="hud-resource-btn" title="Line Token">
          <svg width="30" height="22" viewBox="0 0 30 22" fill="none">
            <rect x="3" y="8" width="24" height="6" rx="3" fill="#e64b3c" />
            <circle cx="6" cy="11" r="3" fill="#ffffff" stroke="#252525" stroke-width="1.5" />
            <circle cx="24" cy="11" r="3" fill="#ffffff" stroke="#252525" stroke-width="1.5" />
          </svg>
          <span id="hud-line-count" class="hud-badge">0</span>
        </div>

        <div id="hud-train-token" class="hud-resource-btn" title="Locomotive">
          <svg width="30" height="22" viewBox="0 0 30 22" fill="none">
            <rect x="3" y="6" width="24" height="10" rx="3" fill="#ffffff" stroke="currentColor" stroke-width="3" />
            <circle cx="9" cy="17" r="1.7" fill="currentColor" />
            <circle cx="21" cy="17" r="1.7" fill="currentColor" />
          </svg>
          <span id="hud-train-count" class="hud-badge">0</span>
        </div>

        <div id="hud-carriage-token" class="hud-resource-btn" title="Carriage">
          <svg width="30" height="22" viewBox="0 0 30 22" fill="none">
            <rect x="4" y="7" width="22" height="9" rx="3" fill="#ffffff" stroke="currentColor" stroke-width="3" />
          </svg>
          <span id="hud-carriage-count" class="hud-badge">0</span>
        </div>

        <div id="hud-tunnel-token" class="hud-resource-btn" title="Tunnel">
          <svg width="30" height="22" viewBox="0 0 30 22" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">
            <path d="M6 17V12C6 7 10 4 15 4C20 4 24 7 24 12V17" />
            <line x1="4" y1="17" x2="26" y2="17" />
          </svg>
          <span id="hud-tunnel-count" class="hud-badge">0</span>
        </div>

        <div id="hud-interchange-token" class="hud-resource-btn" title="Interchange Hub">
          <svg width="30" height="22" viewBox="0 0 30 22" fill="none">
            <circle cx="15" cy="11" r="6" fill="#ffffff" stroke="currentColor" stroke-width="2.5" />
            <circle cx="15" cy="11" r="9" stroke="currentColor" stroke-width="1.8" stroke-dasharray="3 2" />
          </svg>
          <span id="hud-interchange-count" class="hud-badge">0</span>
        </div>
      </div>

      <!-- Weekly Reward Choice Modal -->
      <div id="hud-reward-modal" class="hud-modal-overlay hidden">
        <div class="hud-modal-card">
          <h2>End of week</h2>
          <p>Choose one upgrade</p>
          <div id="hud-reward-options" class="hud-reward-options"></div>
        </div>
      </div>

      <!-- Game Over Modal -->
      <div id="hud-gameover-modal" class="hud-modal-overlay hidden">
        <div class="hud-modal-card gameover">
          <h1>Game over</h1>
          <p class="gameover-sub">Station overcrowded</p>
          <div class="gameover-score-box">
            <span class="score-num" id="gameover-score">0</span>
            <span class="score-lbl">Passengers</span>
          </div>
          <button id="hud-restart-btn" class="hud-primary-btn">Play Again</button>
        </div>
      </div>

      <!-- Toast Container -->
      <div id="hud-toast" class="hud-toast hidden"></div>
    `;

    this.dayText = document.getElementById('hud-day-text')!;
    this.scoreText = document.getElementById('hud-score-text')!;
    this.clockHand = document.getElementById('hud-clock-hand') as any;

    this.pauseBtn = document.getElementById('hud-pause-btn') as HTMLButtonElement;
    this.playBtn = document.getElementById('hud-play-btn') as HTMLButtonElement;
    this.fastBtn = document.getElementById('hud-fast-btn') as HTMLButtonElement;

    this.linesStack = document.getElementById('hud-lines-stack')!;
    this.lineTokenBtn = document.getElementById('hud-line-token-btn')!;
    this.lineCount = document.getElementById('hud-line-count')!;
    this.trainToken = document.getElementById('hud-train-token')!;
    this.trainCount = document.getElementById('hud-train-count')!;
    this.carriageToken = document.getElementById('hud-carriage-token')!;
    this.carriageCount = document.getElementById('hud-carriage-count')!;
    this.tunnelToken = document.getElementById('hud-tunnel-token')!;
    this.tunnelCount = document.getElementById('hud-tunnel-count')!;
    this.interchangeToken = document.getElementById('hud-interchange-token')!;
    this.interchangeCount = document.getElementById('hud-interchange-count')!;

    this.rewardModal = document.getElementById('hud-reward-modal')!;
    this.rewardOptions = document.getElementById('hud-reward-options')!;
    this.gameOverModal = document.getElementById('hud-gameover-modal')!;
    this.toastEl = document.getElementById('hud-toast')!;
  }

  private attachEventListeners(): void {
    this.pauseBtn.addEventListener('click', () => {
      this.wsClient.sendAction({ type: 'pause' });
      this.updateSpeedButtons('pause');
    });

    this.playBtn.addEventListener('click', () => {
      this.wsClient.sendAction({ type: 'resume' });
      this.wsClient.sendAction({ type: 'set_speed', payload: { tps: 30 } });
      this.updateSpeedButtons('play');
    });

    this.fastBtn.addEventListener('click', () => {
      this.wsClient.sendAction({ type: 'resume' });
      this.wsClient.sendAction({ type: 'set_speed', payload: { tps: 75 } });
      this.updateSpeedButtons('fast');
    });

    this.lineTokenBtn.addEventListener('mousedown', (event) => {
      if (this.dragHandler && !this.lineTokenBtn.classList.contains('disabled')) {
        const snap = this.wsClient.getSnapshot();
        const activeLines = snap ? (snap.lines || []).filter((l) => !l.removed) : [];
        const activeIds = new Set(activeLines.map((l) => l.id));
        let nextIdx = 0;
        while (activeIds.has(nextIdx)) nextIdx++;
        this.dragHandler.startDragNewLine(nextIdx, { x: event.clientX, y: event.clientY });
      }
    });

    this.trainToken.addEventListener('mousedown', (event) => {
      if (this.dragHandler && !this.trainToken.classList.contains('disabled')) {
        this.dragHandler.startDragTrain({ x: event.clientX, y: event.clientY });
      }
    });

    this.carriageToken.addEventListener('mousedown', (event) => {
      if (this.dragHandler && !this.carriageToken.classList.contains('disabled')) {
        this.dragHandler.startDragCarriage({ x: event.clientX, y: event.clientY });
      }
    });

    this.interchangeToken.addEventListener('mousedown', (event) => {
      if (this.dragHandler && !this.interchangeToken.classList.contains('disabled')) {
        this.dragHandler.startDragInterchange({ x: event.clientX, y: event.clientY });
      }
    });

    document.getElementById('hud-reset-btn')?.addEventListener('click', () => {
      this.wsClient.sendAction({ type: 'restart' });
    });

    document.getElementById('hud-restart-btn')?.addEventListener('click', () => {
      this.wsClient.sendAction({ type: 'restart' });
      this.gameOverModal.classList.add('hidden');
    });

    this.wsClient.onError((msg) => this.showToast(msg));
  }

  private updateSpeedButtons(mode: 'pause' | 'play' | 'fast'): void {
    this.pauseBtn.classList.toggle('active', mode === 'pause');
    this.playBtn.classList.toggle('active', mode === 'play');
    this.fastBtn.classList.toggle('active', mode === 'fast');
  }

  private currentRewardChoicesKey: string = '';

  public updateState(snap: StateSnapshot): void {
    // 1. Score
    this.scoreText.innerText = String(snap.score);

    // 2. Day & Clock (1 day = 100 ticks)
    const dayIdx = Math.floor(snap.tick / 100) % 7;
    this.dayText.innerText = DAYS[dayIdx];

    const clockAngle = ((snap.tick % 100) / 100) * 360;
    this.clockHand.setAttribute('transform', `rotate(${clockAngle} 18 18)`);

    // 3. Speed status
    if (snap.paused) {
      this.updateSpeedButtons('pause');
    } else if (snap.tps > 40) {
      this.updateSpeedButtons('fast');
    } else {
      this.updateSpeedButtons('play');
    }

    // 4. Resources Dock
    const res = (snap.resources || {}) as any;
    const availableLinesCount = res.lines ?? res.Lines ?? 0;
    const trains = res.trains ?? res.Trains ?? 0;
    const carriages = res.carriages ?? res.Carriages ?? 0;
    const tunnels = res.tunnels ?? res.Tunnels ?? 0;
    const interchanges = res.interchanges ?? res.Interchanges ?? 0;

    this.lineCount.innerText = String(availableLinesCount);
    this.trainCount.innerText = String(trains);
    this.carriageCount.innerText = String(carriages);
    this.tunnelCount.innerText = String(tunnels);
    this.interchangeCount.innerText = String(interchanges);

    this.lineTokenBtn.classList.toggle('disabled', availableLinesCount <= 0);
    this.trainToken.classList.toggle('disabled', trains <= 0);
    this.carriageToken.classList.toggle('disabled', carriages <= 0);
    this.tunnelToken.classList.toggle('disabled', tunnels <= 0);
    this.interchangeToken.classList.toggle('disabled', interchanges <= 0);

    // 5. Line Inventory Stack
    this.renderLineStack(snap);

    // 6. Weekly Reward Modal
    if (snap.pending_reward_choices && snap.pending_reward_choices.length > 0) {
      const choicesKey = snap.pending_reward_choices.join(',');
      if (choicesKey !== this.currentRewardChoicesKey) {
        this.currentRewardChoicesKey = choicesKey;
        this.showRewardModal(snap.pending_reward_choices);
      }
      this.rewardModal.classList.remove('hidden');
    } else {
      this.currentRewardChoicesKey = '';
      this.rewardModal.classList.add('hidden');
    }

    // 7. Game Over Modal
    if (!snap.alive) {
      document.getElementById('gameover-score')!.innerText = String(snap.score);
      this.gameOverModal.classList.remove('hidden');
    } else {
      this.gameOverModal.classList.add('hidden');
    }
  }

  private renderLineStack(snap: StateSnapshot): void {
    this.linesStack.innerHTML = '';
    const res = (snap.resources || {}) as any;
    const availableLinesCount = res.lines ?? res.Lines ?? 0;
    const activeLines = (snap.lines || []).filter((l) => !l.removed);
    const activeIds = new Set(activeLines.map((l) => l.id));

    // 1. Active Lines (Click to delete line & refund resources)
    for (const line of activeLines) {
      const color = getLineColor(line.id);
      const token = document.createElement('div');
      token.className = 'hud-line-token used';
      token.style.setProperty('--line-color', color);
      token.title = `Click to delete Line ${line.id + 1} (${line.stations.length} stations)`;

      token.addEventListener('click', () => {
        console.log(`🗑️ [FRONTEND] Removing Line ${line.id}`);
        this.wsClient.sendAction({
          type: 'remove_line',
          payload: { line_id: line.id },
        });
      });

      this.linesStack.appendChild(token);
    }

    // 2. Available Lines (Draggable to create line)
    let nextIdx = 0;
    for (let i = 0; i < availableLinesCount; i++) {
      while (activeIds.has(nextIdx)) {
        nextIdx++;
      }
      const lineIndexToUse = nextIdx;
      nextIdx++;

      const color = getLineColor(lineIndexToUse);
      const token = document.createElement('div');
      token.className = 'hud-line-token available';
      token.style.setProperty('--line-color', color);
      token.title = `Available Line (Color: ${color})`;

      token.addEventListener('mousedown', (e) => {
        if (this.dragHandler) {
          this.dragHandler.startDragNewLine(lineIndexToUse, { x: e.clientX, y: e.clientY });
        }
      });

      this.linesStack.appendChild(token);
    }
  }

  private showRewardModal(choices: RewardType[]): void {
    this.rewardOptions.innerHTML = '';
    choices.forEach((choice) => {
      const card = document.createElement('div');
      card.className = 'hud-reward-card';

      const label = REWARD_LABELS[choice] || 'Upgrade';
      card.innerHTML = `
        <div class="reward-icon">${this.getRewardIconSvg(choice)}</div>
        <span class="reward-title">${label}</span>
      `;

      card.addEventListener('click', () => {
        this.wsClient.sendAction({
          type: 'choose_reward',
          payload: { choice: choice },
        });
        this.rewardModal.classList.add('hidden');
      });

      this.rewardOptions.appendChild(card);
    });
  }

  private getRewardIconSvg(type: RewardType): string {
    switch (type) {
      case 0: // Line
        return `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#252525" stroke-width="3"><line x1="4" y1="12" x2="20" y2="12"/><circle cx="4" cy="12" r="3" fill="#ffffff"/><circle cx="20" cy="12" r="3" fill="#ffffff"/></svg>`;
      case 1: // Train
        return `<svg width="32" height="32" viewBox="0 0 30 22" fill="none"><rect x="3" y="6" width="24" height="10" rx="3" fill="#ffffff" stroke="#252525" stroke-width="3"/><circle cx="9" cy="17" r="1.7" fill="#252525"/><circle cx="21" cy="17" r="1.7" fill="#252525"/></svg>`;
      case 2: // Tunnel
        return `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#252525" stroke-width="3" stroke-linecap="round"><path d="M6 17V12C6 7 10 4 15 4C20 4 24 7 24 12V17"/><line x1="4" y1="17" x2="26" y2="17"/></svg>`;
      case 3: // Carriage
        return `<svg width="32" height="32" viewBox="0 0 30 22" fill="none"><rect x="4" y="7" width="22" height="9" rx="3" fill="#ffffff" stroke="#252525" stroke-width="3"/></svg>`;
      case 4: // Interchange
        return `<svg width="32" height="32" viewBox="0 0 30 22" fill="none"><circle cx="15" cy="11" r="6" fill="#ffffff" stroke="#252525" stroke-width="2.5"/><circle cx="15" cy="11" r="9" stroke="#252525" stroke-width="1.8" stroke-dasharray="3 2"/></svg>`;
      default:
        return ``;
    }
  }

  private showToast(message: string): void {
    console.warn(`[HUD Toast] ${message}`);
    this.toastEl.innerText = message;
    this.toastEl.classList.remove('hidden');
    setTimeout(() => {
      this.toastEl.classList.add('hidden');
    }, 2800);
  }
}

export { HUD as HUDManager };
