import './style.css';
import { GameWSClient } from './ws/client';
import { Viewport } from './renderer/viewport';
import { renderMapWater } from './renderer/map';
import { renderMetroLines } from './renderer/lines';
import { renderStations } from './renderer/stations';
import { TrainInterpolator } from './renderer/trains';
import { DragHandler } from './interaction/dragHandler';
import { HUDManager } from './ui/hud';
import type { StateSnapshot } from './types';

class MiniMetroApp {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private viewport: Viewport = new Viewport();
  private wsClient: GameWSClient;
  private trainInterpolator: TrainInterpolator = new TrainInterpolator();
  private dragHandler!: DragHandler;
  private hudManager!: HUDManager;
  private latestSnapshot: StateSnapshot | null = null;

  constructor() {
    this.wsClient = new GameWSClient();
    this.initDOM();
    this.initCanvas();
    this.initEngine();
  }

  private initDOM(): void {
    const appEl = document.getElementById('app')!;
    appEl.innerHTML = `
      <canvas id="game-canvas"></canvas>
      <div id="hud-overlay"></div>
    `;

    this.canvas = document.getElementById('game-canvas') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d')!;

    const hudEl = document.getElementById('hud-overlay')!;
    this.hudManager = new HUDManager(hudEl, this.wsClient);
  }

  private initCanvas(): void {
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  private resizeCanvas(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;

    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.viewport.updateSize(w, h);
  }

  private initEngine(): void {
    this.dragHandler = new DragHandler(this.canvas, this.viewport, this.wsClient);
    this.hudManager.setDragHandler(this.dragHandler);

    this.wsClient.onState((snapshot) => {
      this.latestSnapshot = snapshot;
      this.hudManager.updateState(snapshot);
    });

    this.wsClient.connect();

    requestAnimationFrame(this.renderLoop.bind(this));
  }

  private renderLoop(): void {
    this.render();
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  private render(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.ctx.fillStyle = '#f5f2ec';
    this.ctx.fillRect(0, 0, w, h);

    if (!this.latestSnapshot) {
      this.renderConnectingScreen(w, h);
      return;
    }

    const snap = this.latestSnapshot;

    renderMapWater(this.ctx, this.viewport, snap.rivers, snap.water_polygons);

    const activeDragPreview = this.dragHandler.getActiveDragPreview();
    renderMetroLines(
      this.ctx,
      this.viewport,
      snap.lines || [],
      snap.stations || [],
      activeDragPreview || undefined,
      snap.rivers || []
    );

    const hoveredStId = this.dragHandler.getHoveredStationId();
    renderStations(
      this.ctx,
      this.viewport,
      snap.stations || [],
      hoveredStId
    );

    this.trainInterpolator.renderTrains(
      this.ctx,
      this.viewport,
      snap.trains || [],
      snap.lines || [],
      snap.stations || []
    );

    this.dragHandler.renderDragOverlay(this.ctx);
  }

  private renderConnectingScreen(w: number, h: number): void {
    this.ctx.save();
    this.ctx.fillStyle = '#252525';
    this.ctx.font = '600 18px Avenir Next, Inter, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText('Connecting to simulator...', w / 2, h / 2);
    this.ctx.restore();
  }
}

new MiniMetroApp();
