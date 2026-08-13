import type { StateSnapshot } from '../types';
import { RewardType } from '../types';
import { GameWSClient } from '../ws/client';
import { LINE_COLORS } from '../renderer/shapes';
import { DragHandler } from '../interaction/dragHandler';

const DAYS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];

export class HUDManager {
  private container: HTMLElement;
  private wsClient: GameWSClient;
  private dragHandler: DragHandler | null = null;

  // DOM Elements
  private dayText!: HTMLElement;
  private scoreText!: HTMLElement;
  private clockHand!: SVGLineElement;
  private pauseBtn!: HTMLElement;
  private playBtn!: HTMLElement;
  private fastBtn!: HTMLElement;

  private linesStack!: HTMLElement;
  private trainToken!: HTMLElement;
  private trainCount!: HTMLElement;
  private carriageToken!: HTMLElement;
  private carriageCount!: HTMLElement;
  private tunnelToken!: HTMLElement;
  private tunnelCount!: HTMLElement;

  private rewardModal!: HTMLElement;
  private gameOverModal!: HTMLElement;
  private errorToast!: HTMLElement;

  constructor(container: HTMLElement, wsClient: GameWSClient) {
    this.container = container;
    this.wsClient = wsClient;

    this.createDomElements();
    this.attachListeners();
  }

  public setDragHandler(handler: DragHandler): void {
    this.dragHandler = handler;
  }

  private createDomElements(): void {
    this.container.innerHTML = `
      <!-- Top Left Bar -->
      <div class="hud-top-left">
        <button id="hud-back-btn" class="hud-circle-btn" title="Back to Menu">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="19" y1="12" x2="5" y2="12"></line>
            <polyline points="12 19 5 12 12 5"></polyline>
          </svg>
        </button>
      </div>

      <!-- Top Right Bar (Day, Score, Analog Clock) -->
      <div class="hud-top-right">
        <span id="hud-day-text" class="hud-day">SAT</span>
        <span id="hud-score-text" class="hud-score">0</span>
        <div class="hud-clock-container">
          <svg id="hud-clock-svg" width="34" height="34" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="16" fill="#e63946" stroke="#22252a" stroke-width="2.5" />
            <line id="hud-clock-hand" x1="18" y1="18" x2="18" y2="6" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" />
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
          <button id="hud-play-btn" class="hud-speed-btn active" title="1x Speed">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          </button>
          <button id="hud-fast-btn" class="hud-speed-btn" title="2x Fast Forward">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="3 3 13 12 3 21 3 3" />
              <polygon points="13 3 23 12 13 21 13 3" />
            </svg>
          </button>
        </div>

        <div id="hud-lines-stack" class="hud-lines-stack">
        </div>
      </div>

      <!-- Far Left Resource Dock (Train, Carriage, Tunnel) -->
      <div id="hud-left-dock" class="hud-left-dock">
        <div id="hud-train-token" class="hud-resource-btn" title="Train / Locomotive">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="4" y="4" width="16" height="16" rx="2" />
            <circle cx="8" cy="16" r="1.5" />
            <circle cx="16" cy="16" r="1.5" />
            <line x1="8" y1="8" x2="16" y2="8" />
          </svg>
          <span id="hud-train-count" class="hud-badge">0</span>
        </div>

        <div id="hud-carriage-token" class="hud-resource-btn" title="Carriage">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="6" width="18" height="12" rx="2" />
            <circle cx="7" cy="18" r="1.5" />
            <circle cx="17" cy="18" r="1.5" />
          </svg>
          <span id="hud-carriage-count" class="hud-badge">0</span>
        </div>

        <div id="hud-tunnel-token" class="hud-resource-btn" title="Bridge / Tunnel">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 18V12C4 7.58172 7.58172 4 12 4C16.4183 4 20 7.58172 20 12V18" />
            <line x1="2" y1="18" x2="22" y2="18" />
          </svg>
          <span id="hud-tunnel-count" class="hud-badge">0</span>
        </div>
      </div>

      <!-- Weekly Reward Choice Modal -->
      <div id="hud-reward-modal" class="hud-modal-overlay hidden">
        <div class="hud-modal-card">
          <h2>Choose Weekly Upgrade</h2>
          <p>Select one asset to add to your metro network</p>
          <div id="hud-reward-options" class="hud-reward-options"></div>
        </div>
      </div>

      <!-- Game Over Modal -->
      <div id="hud-gameover-modal" class="hud-modal-overlay hidden">
        <div class="hud-modal-card gameover">
          <h1>Network Closed</h1>
          <p class="gameover-sub">Station Overcrowded</p>
          <div class="gameover-score-box">
            <span class="score-num" id="gameover-score">0</span>
            <span class="score-lbl">Passengers Transported</span>
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

    this.pauseBtn = document.getElementById('hud-pause-btn')!;
    this.playBtn = document.getElementById('hud-play-btn')!;
    this.fastBtn = document.getElementById('hud-fast-btn')!;

    this.linesStack = document.getElementById('hud-lines-stack')!;
    this.trainToken = document.getElementById('hud-train-token')!;
    this.trainCount = document.getElementById('hud-train-count')!;
    this.carriageToken = document.getElementById('hud-carriage-token')!;
    this.carriageCount = document.getElementById('hud-carriage-count')!;
    this.tunnelToken = document.getElementById('hud-tunnel-token')!;
    this.tunnelCount = document.getElementById('hud-tunnel-count')!;

    this.rewardModal = document.getElementById('hud-reward-modal')!;
    this.gameOverModal = document.getElementById('hud-gameover-modal')!;
    this.errorToast = document.getElementById('hud-toast')!;
  }

  private attachListeners(): void {
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
      this.wsClient.sendAction({ type: 'set_speed', payload: { tps: 60 } });
      this.updateSpeedButtons('fast');
    });

    this.trainToken.addEventListener('mousedown', () => {
      if (this.dragHandler) this.dragHandler.startDragTrain();
    });

    this.carriageToken.addEventListener('mousedown', () => {
      if (this.dragHandler) this.dragHandler.startDragCarriage();
    });

    document.getElementById('hud-restart-btn')?.addEventListener('click', () => {
      window.location.reload();
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
    this.trainCount.innerText = String(snap.resources.trains || 0);
    this.carriageCount.innerText = String(snap.resources.carriages || 0);
    this.tunnelCount.innerText = String(snap.resources.tunnels || 0);

    this.trainToken.classList.toggle('disabled', snap.resources.trains <= 0);
    this.carriageToken.classList.toggle('disabled', snap.resources.carriages <= 0);
    this.tunnelToken.classList.toggle('disabled', snap.resources.tunnels <= 0);

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
    const totalLinesAvailable = (snap.lines ? snap.lines.filter((l) => !l.removed).length : 0) + (snap.resources.lines || 0);
    const maxVisibleLines = Math.max(3, totalLinesAvailable);

    let html = '';
    for (let i = 0; i < maxVisibleLines; i++) {
      const color = LINE_COLORS[i % LINE_COLORS.length];
      const isUsed = snap.lines && snap.lines.some((l) => l.id === i && !l.removed);
      const isAvailable = !isUsed && i < totalLinesAvailable;

      const classes = ['hud-line-circle'];
      if (isUsed) classes.push('used');
      if (isAvailable) classes.push('available');

      html += `<div class="${classes.join(' ')}" style="background-color: ${color};" data-line-index="${i}"></div>`;
    }

    this.linesStack.innerHTML = html;

    const circles = this.linesStack.querySelectorAll('.hud-line-circle.available');
    circles.forEach((el) => {
      el.addEventListener('mousedown', () => {
        const idx = parseInt(el.getAttribute('data-line-index') || '0', 10);
        if (this.dragHandler) {
          const rect = el.getBoundingClientRect();
          this.dragHandler.startDragNewLine(idx, { x: rect.left + 15, y: rect.top + 15 });
        }
      });
    });
  }

  private showRewardModal(choices: RewardType[]): void {
    this.rewardModal.classList.remove('hidden');
    const optsContainer = document.getElementById('hud-reward-options')!;

    const labels: Record<number, { title: string; icon: string }> = {
      [RewardType.RewardLine]: { title: 'New Line', icon: '🚇' },
      [RewardType.RewardTrain]: { title: 'Locomotive', icon: '🚂' },
      [RewardType.RewardTunnel]: { title: 'Bridge / Tunnel', icon: '🌉' },
      [RewardType.RewardCarriage]: { title: 'Carriage', icon: '🚃' },
      [RewardType.RewardInterchange]: { title: 'Interchange Hub', icon: '🏢' },
    };

    let html = '';
    for (const choice of choices) {
      const info = labels[choice] || { title: 'Upgrade', icon: '🎁' };
      html += `
        <button class="hud-reward-card" data-choice="${choice}">
          <span class="hud-reward-icon">${info.icon}</span>
          <span class="hud-reward-title">${info.title}</span>
        </button>
      `;
    }
    optsContainer.innerHTML = html;

    const cards = optsContainer.querySelectorAll('.hud-reward-card');
    cards.forEach((card) => {
      card.addEventListener('click', () => {
        const choice = parseInt(card.getAttribute('data-choice') || '0', 10);
        this.wsClient.sendAction({
          type: 'choose_reward',
          payload: { choice },
        });
        this.rewardModal.classList.add('hidden');
      });
    });
  }

  private showToast(msg: string): void {
    this.errorToast.innerText = msg;
    this.errorToast.classList.remove('hidden');
    setTimeout(() => {
      this.errorToast.classList.add('hidden');
    }, 3000);
  }
}
