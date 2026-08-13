import type { StationDTO, Pos, DragState } from '../types';
import { getX, getY } from '../types';
import { Viewport } from '../renderer/viewport';
import { GameWSClient } from '../ws/client';
import { getLineColor } from '../renderer/lines';

export class DragHandler {
  private canvas: HTMLCanvasElement;
  private viewport: Viewport;
  private wsClient: GameWSClient;
  private dragState: DragState | null = null;
  private hoveredStationId: number | null = null;

  constructor(canvas: HTMLCanvasElement, viewport: Viewport, wsClient: GameWSClient) {
    this.canvas = canvas;
    this.viewport = viewport;
    this.wsClient = wsClient;

    this.attachEvents();
  }

  private attachEvents(): void {
    this.canvas.addEventListener('mousedown', this.onPointerDown.bind(this));
    window.addEventListener('mousemove', this.onPointerMove.bind(this));
    window.addEventListener('mouseup', this.onPointerUp.bind(this));

    this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: false });
    window.addEventListener('touchmove', this.onTouchMove.bind(this), { passive: false });
    window.addEventListener('touchend', this.onTouchEnd.bind(this));
  }

  private getEventPos(e: MouseEvent | Touch): Pos {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  public getHoveredStationId(): number | null {
    return this.hoveredStationId;
  }

  public getActiveDragPreview(): { points: Pos[]; color: string } | null {
    if (!this.dragState) return null;

    const snap = this.wsClient.getSnapshot();
    if (!snap) return null;

    let points: Pos[] = [];
    let color = '#22252a';

    if (this.dragState.source.type === 'new_line') {
      color = getLineColor(this.dragState.source.lineIndex);
    } else if (this.dragState.source.type === 'extend_line') {
      const line = snap.lines.find((l) => l.id === (this.dragState!.source as any).lineId);
      if (line) color = getLineColor(line.id);

      const fromSt = snap.stations.find((s) => s.id === (this.dragState!.source as any).fromStationId);
      if (fromSt) {
        points.push(this.viewport.mapToScreen({ x: fromSt.x, y: fromSt.y }));
      }
    }

    const currentScreen = this.dragState.targetStationId !== null
      ? (() => {
          const st = snap.stations.find((s) => s.id === this.dragState!.targetStationId);
          return st ? this.viewport.mapToScreen({ x: st.x, y: st.y }) : this.dragState!.currentPos;
        })()
      : this.dragState.currentPos;

    points.push(currentScreen);

    return { points, color };
  }

  private findStationAt(pos: Pos, stations: StationDTO[]): StationDTO | null {
    const snapThreshold = 25;
    const px = getX(pos);
    const py = getY(pos);

    for (const st of stations) {
      if (!st.alive) continue;
      const screenP = this.viewport.mapToScreen({ x: st.x, y: st.y });
      const dx = screenP.x - px;
      const dy = screenP.y - py;
      if (Math.sqrt(dx * dx + dy * dy) <= snapThreshold) {
        return st;
      }
    }
    return null;
  }

  public startDragNewLine(lineIndex: number, startPos: Pos): void {
    this.dragState = {
      source: { type: 'new_line', lineIndex },
      currentPos: startPos,
      targetStationId: null,
    };
  }

  public startDragTrain(): void {
    this.dragState = {
      source: { type: 'add_train' },
      currentPos: { x: this.viewport.width / 2, y: this.viewport.height / 2 },
      targetStationId: null,
    };
  }

  public startDragCarriage(): void {
    this.dragState = {
      source: { type: 'add_carriage' },
      currentPos: { x: this.viewport.width / 2, y: this.viewport.height / 2 },
      targetStationId: null,
    };
  }

  private onPointerDown(e: MouseEvent): void {
    const pos = this.getEventPos(e);
    const snap = this.wsClient.getSnapshot();
    if (!snap) return;

    const st = this.findStationAt(pos, snap.stations);
    if (!st) return;

    for (const line of snap.lines) {
      if (line.removed || !line.stations || line.stations.length === 0) continue;

      const firstStId = line.stations[0];
      const lastStId = line.stations[line.stations.length - 1];

      if (st.id === lastStId) {
        this.dragState = {
          source: { type: 'extend_line', lineId: line.id, fromStationId: lastStId, fromFront: false },
          currentPos: pos,
          targetStationId: null,
        };
        return;
      }

      if (st.id === firstStId) {
        this.dragState = {
          source: { type: 'extend_line', lineId: line.id, fromStationId: firstStId, fromFront: true },
          currentPos: pos,
          targetStationId: null,
        };
        return;
      }
    }

    if (snap.resources.lines > 0) {
      this.dragState = {
        source: { type: 'extend_line', lineId: -1, fromStationId: st.id, fromFront: false },
        currentPos: pos,
        targetStationId: null,
      };
    }
  }

  private onPointerMove(e: MouseEvent): void {
    const pos = this.getEventPos(e);
    const snap = this.wsClient.getSnapshot();
    if (!snap) return;

    const st = this.findStationAt(pos, snap.stations);
    this.hoveredStationId = st ? st.id : null;

    if (this.dragState) {
      this.dragState.currentPos = pos;
      this.dragState.targetStationId = this.hoveredStationId;
    }
  }

  private onPointerUp(): void {
    if (!this.dragState) return;

    const snap = this.wsClient.getSnapshot();
    const targetStId = this.dragState.targetStationId;
    const source = this.dragState.source;

    if (snap && targetStId !== null) {
      if (source.type === 'extend_line') {
        if (source.lineId === -1) {
          if (source.fromStationId !== targetStId) {
            this.wsClient.sendAction({
              type: 'add_line',
              payload: { stations: [source.fromStationId, targetStId] },
            });
          }
        } else {
          const line = snap.lines.find((l) => l.id === source.lineId);
          if (line && line.stations.length > 2 && line.stations[0] === targetStId) {
            this.wsClient.sendAction({
              type: 'close_loop',
              payload: { line_id: line.id, use_tunnel: false },
            });
          } else if (source.fromStationId !== targetStId) {
            this.wsClient.sendAction({
              type: 'extend_line',
              payload: { line_id: source.lineId, station_id: targetStId, use_tunnel: false },
            });
          }
        }
      } else if (source.type === 'add_train') {
        const line = snap.lines.find((l) => !l.removed && l.stations.includes(targetStId));
        if (line) {
          this.wsClient.sendAction({
            type: 'add_train',
            payload: { line_id: line.id },
          });
        }
      } else if (source.type === 'add_carriage') {
        const train = snap.trains && snap.trains.length > 0 ? snap.trains[0] : null;
        if (train) {
          this.wsClient.sendAction({
            type: 'add_carriage',
            payload: { train_id: train.id },
          });
        }
      }
    }

    this.dragState = null;
  }

  private onTouchStart(e: TouchEvent): void {
    if (e.touches.length > 0) {
      e.preventDefault();
      this.onPointerDown(e.touches[0] as any);
    }
  }

  private onTouchMove(e: TouchEvent): void {
    if (e.touches.length > 0) {
      e.preventDefault();
      this.onPointerMove(e.touches[0] as any);
    }
  }

  private onTouchEnd(): void {
    this.onPointerUp();
  }
}
