import type { StationDTO, LineDTO, Pos, DragState } from '../types';
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
  private selectedStationId: number | null = null;

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
    const snap = this.wsClient.getSnapshot();
    if (!snap) return null;

    const stations = snap.stations || [];
    const lines = snap.lines || [];

    // Click-to-connect active preview
    if (this.selectedStationId !== null && !this.dragState) {
      const st = stations.find((s) => s.id === this.selectedStationId);
      if (st) {
        const p1 = this.viewport.mapToScreen({ x: getX(st), y: getY(st) });
        const p2 = this.hoveredStationId !== null
          ? (() => {
              const hSt = stations.find((s) => s.id === this.hoveredStationId);
              return hSt ? this.viewport.mapToScreen({ x: getX(hSt), y: getY(hSt) }) : p1;
            })()
          : p1;
        return { points: [p1, p2], color: '#f53649' };
      }
    }

    if (!this.dragState) return null;

    let points: Pos[] = [];
    let color = '#22252a';

    if (this.dragState.source.type === 'new_line') {
      color = getLineColor(this.dragState.source.lineIndex);
      const firstStId = (this.dragState.source as any).firstStationId;
      if (firstStId !== null && firstStId !== undefined) {
        const st = stations.find((s) => s.id === firstStId);
        if (st) {
          points.push(this.viewport.mapToScreen({ x: getX(st), y: getY(st) }));
        }
      }
    } else if (this.dragState.source.type === 'extend_line') {
      const line = lines.find((l) => l.id === (this.dragState!.source as any).lineId);
      if (line) color = getLineColor(line.id);

      const fromSt = stations.find((s) => s.id === (this.dragState!.source as any).fromStationId);
      if (fromSt) {
        points.push(this.viewport.mapToScreen({ x: getX(fromSt), y: getY(fromSt) }));
      }
    }

    const currentScreen = this.dragState.targetStationId !== null
      ? (() => {
          const st = stations.find((s) => s.id === this.dragState!.targetStationId);
          return st ? this.viewport.mapToScreen({ x: getX(st), y: getY(st) }) : this.dragState!.currentPos;
        })()
      : this.dragState.currentPos;

    if (points.length > 0) {
      points.push(currentScreen);
    }

    return points.length >= 2 ? { points, color } : null;
  }

  private findStationAt(pos: Pos, stations: StationDTO[], threshold: number = 55): StationDTO | null {
    if (!stations) return null;
    const px = getX(pos);
    const py = getY(pos);

    let closest: StationDTO | null = null;
    let minDist = threshold;

    for (const st of stations) {
      if (!st.alive) continue;
      const screenP = this.viewport.mapToScreen({ x: getX(st), y: getY(st) });
      const dx = screenP.x - px;
      const dy = screenP.y - py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= minDist) {
        minDist = dist;
        closest = st;
      }
    }
    return closest;
  }

  private findTerminalToExtend(pos: Pos, lines: LineDTO[], stations: StationDTO[]): { lineId: number; stationId: number; fromFront: boolean } | null {
    if (!lines || !stations) return null;
    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) stationMap.set(st.id, st);

    const px = getX(pos), py = getY(pos);
    const threshold = 40; // 40px terminal end cap hit box

    for (const line of lines) {
      if (line.removed || !line.stations || line.stations.length === 0) continue;

      const firstStId = line.stations[0];
      const lastStId = line.stations[line.stations.length - 1];

      const firstSt = stationMap.get(firstStId);
      const lastSt = stationMap.get(lastStId);

      if (lastSt) {
        const lastP = this.viewport.mapToScreen({ x: getX(lastSt), y: getY(lastSt) });
        const dx = lastP.x - px, dy = lastP.y - py;
        if (Math.sqrt(dx * dx + dy * dy) <= threshold) {
          return { lineId: line.id, stationId: lastStId, fromFront: false };
        }
      }

      if (firstSt) {
        const firstP = this.viewport.mapToScreen({ x: getX(firstSt), y: getY(firstSt) });
        const dx = firstP.x - px, dy = firstP.y - py;
        if (Math.sqrt(dx * dx + dy * dy) <= threshold) {
          return { lineId: line.id, stationId: firstStId, fromFront: true };
        }
      }
    }

    return null;
  }

  private findLineNear(pos: Pos, lines: LineDTO[], stations: StationDTO[]): LineDTO | null {
    if (!lines || !stations) return null;
    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) stationMap.set(st.id, st);

    const threshold = 35;
    const px = getX(pos);
    const py = getY(pos);

    for (const line of lines) {
      if (line.removed || !line.stations || line.stations.length < 2) continue;

      for (let i = 0; i < line.stations.length - 1; i++) {
        const st1 = stationMap.get(line.stations[i]);
        const st2 = stationMap.get(line.stations[i + 1]);
        if (!st1 || !st2) continue;

        const p1 = this.viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
        const p2 = this.viewport.mapToScreen({ x: getX(st2), y: getY(st2) });

        const dist = distToSegment({ x: px, y: py }, p1, p2);
        if (dist <= threshold) {
          return line;
        }
      }
    }
    return null;
  }

  public startDragNewLine(lineIndex: number, startPos: Pos): void {
    console.log(`✨ [FRONTEND] Drag started for New Line index ${lineIndex} from token`);
    this.dragState = {
      source: { type: 'new_line', lineIndex, firstStationId: null } as any,
      currentPos: startPos,
      targetStationId: null,
    };
  }

  public startDragTrain(): void {
    console.log('🚂 [FRONTEND] Drag started for Locomotive/Train token');
    this.dragState = {
      source: { type: 'add_train' },
      currentPos: { x: this.viewport.width / 2, y: this.viewport.height / 2 },
      targetStationId: null,
    };
  }

  public startDragCarriage(): void {
    console.log('🚃 [FRONTEND] Drag started for Carriage token');
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

    const stations = snap.stations || [];
    const lines = snap.lines || [];

    // 1. Check line terminal extension hit box
    const terminalHit = this.findTerminalToExtend(pos, lines, stations);
    if (terminalHit) {
      console.log(`✏️ [FRONTEND] Extending Line ${terminalHit.lineId} from station ${terminalHit.stationId} (front=${terminalHit.fromFront})`);
      this.dragState = {
        source: { type: 'extend_line', lineId: terminalHit.lineId, fromStationId: terminalHit.stationId, fromFront: terminalHit.fromFront },
        currentPos: pos,
        targetStationId: null,
      };
      this.selectedStationId = null;
      return;
    }

    const st = this.findStationAt(pos, stations, 55);
    if (!st) {
      this.selectedStationId = null;
      return;
    }

    console.log(`🖱️ [FRONTEND] Mouse down at station ID ${st.id} (${st.kind_name})`);

    // Click-to-connect support
    if (this.selectedStationId !== null && this.selectedStationId !== st.id) {
      const srcStId = this.selectedStationId;
      this.selectedStationId = null;

      // Check if srcStId is a terminal of an existing line
      for (const line of lines) {
        if (line.removed || !line.stations || line.stations.length === 0) continue;
        if (line.stations[0] === srcStId) {
          console.log(`✏️ [FRONTEND] Extending Line ${line.id} from front station ${srcStId} -> ${st.id}`);
          this.wsClient.sendAction({
            type: 'extend_line',
            payload: { line_id: line.id, station_id: st.id, use_tunnel: false, from_front: true },
          });
          return;
        }
        if (line.stations[line.stations.length - 1] === srcStId) {
          console.log(`✏️ [FRONTEND] Extending Line ${line.id} from end station ${srcStId} -> ${st.id}`);
          this.wsClient.sendAction({
            type: 'extend_line',
            payload: { line_id: line.id, station_id: st.id, use_tunnel: false, from_front: false },
          });
          return;
        }
      }

      // Otherwise create a new line
      console.log(`🔗 [FRONTEND] Creating new line: [${srcStId}, ${st.id}]`);
      this.wsClient.sendAction({
        type: 'add_line',
        payload: { stations: [srcStId, st.id] },
      });
      return;
    }

    this.selectedStationId = st.id;

    // Draft a new line from station if unused line tokens exist
    const res = snap.resources || { lines: 0 };
    if (res.lines > 0) {
      console.log(`✏️ [FRONTEND] Starting new line draft from Station ${st.id}`);
      this.dragState = {
        source: { type: 'extend_line', lineId: -1, fromStationId: st.id, fromFront: false },
        currentPos: pos,
        targetStationId: null,
      };
    } else {
      console.warn('⚠️ [FRONTEND] No available lines in resource pool');
    }
  }

  private onPointerMove(e: MouseEvent): void {
    const pos = this.getEventPos(e);
    const snap = this.wsClient.getSnapshot();
    if (!snap) return;

    const stations = snap.stations || [];
    const st = this.findStationAt(pos, stations, 55);
    this.hoveredStationId = st ? st.id : null;

    if (this.dragState) {
      this.dragState.currentPos = pos;
      this.dragState.targetStationId = this.hoveredStationId;

      if (this.dragState.source.type === 'new_line') {
        const src = this.dragState.source as any;
        if (st && src.firstStationId === null) {
          console.log(`📌 [FRONTEND] New line token snapped to first station ${st.id}`);
          src.firstStationId = st.id;
        } else if (st && src.firstStationId !== null && st.id !== src.firstStationId) {
          this.dragState.targetStationId = st.id;
        }
      }
    }
  }

  private onPointerUp(): void {
    if (!this.dragState) return;

    const snap = this.wsClient.getSnapshot();
    const currentPos = this.dragState.currentPos;
    const stations = snap ? snap.stations || [] : [];
    const lines = snap ? snap.lines || [] : [];
    const trains = snap ? snap.trains || [] : [];

    const endSt = this.findStationAt(currentPos, stations, 60);
    const targetStId = endSt ? endSt.id : this.dragState.targetStationId;
    const source = this.dragState.source;

    console.log('👆 [FRONTEND] Mouse up - Action check:', source, 'Target station:', targetStId);

    if (snap) {
      if (source.type === 'new_line') {
        const src = source as any;
        const firstStId = src.firstStationId;
        const secondStId = targetStId;

        if (firstStId !== null && secondStId !== null && firstStId !== secondStId) {
          console.log(`📤 [FRONTEND] Dispatching add_line from Token: [${firstStId}, ${secondStId}]`);
          this.wsClient.sendAction({
            type: 'add_line',
            payload: { stations: [firstStId, secondStId] },
          });
          this.selectedStationId = null;
        }
      } else if (source.type === 'extend_line') {
        if (targetStId !== null) {
          if (source.lineId === -1) {
            if (source.fromStationId !== targetStId) {
              console.log(`📤 [FRONTEND] Dispatching add_line: [${source.fromStationId}, ${targetStId}]`);
              this.wsClient.sendAction({
                type: 'add_line',
                payload: { stations: [source.fromStationId, targetStId] },
              });
              this.selectedStationId = null;
            }
          } else {
            const line = lines.find((l) => l.id === source.lineId);
            if (line && line.stations.length > 2 && line.stations[0] === targetStId) {
              console.log(`📤 [FRONTEND] Dispatching close_loop for Line ${line.id}`);
              this.wsClient.sendAction({
                type: 'close_loop',
                payload: { line_id: line.id, use_tunnel: false },
              });
              this.selectedStationId = null;
            } else if (source.fromStationId !== targetStId) {
              console.log(`📤 [FRONTEND] Dispatching extend_line for Line ${source.lineId} -> Station ${targetStId}`);
              this.wsClient.sendAction({
                type: 'extend_line',
                payload: { line_id: source.lineId, station_id: targetStId, use_tunnel: false, from_front: source.fromFront },
              });
              this.selectedStationId = null;
            }
          }
        }
      } else if (source.type === 'add_train') {
        let line = targetStId !== null
          ? lines.find((l) => !l.removed && l.stations.includes(targetStId))
          : this.findLineNear(currentPos, lines, stations);

        if (line) {
          console.log(`📤 [FRONTEND] Dispatching add_train to Line ${line.id}`);
          this.wsClient.sendAction({
            type: 'add_train',
            payload: { line_id: line.id },
          });
        } else {
          console.warn('⚠️ [FRONTEND] Cannot add train: Drop target is not on any active metro line');
        }
      } else if (source.type === 'add_carriage') {
        const train = trains.length > 0 ? trains[0] : null;
        if (train) {
          console.log(`📤 [FRONTEND] Dispatching add_carriage to Train ${train.id}`);
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

function distToSegment(p: Pos, v: Pos, w: Pos): number {
  const px = getX(p), py = getY(p);
  const vx = getX(v), vy = getY(v);
  const wx = getX(w), wy = getY(w);

  const l2 = (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy);
  if (l2 === 0) return Math.sqrt((px - vx) * (px - vx) + (py - vy) * (py - vy));

  let t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
  t = Math.max(0, Math.min(1, t));

  const projX = vx + t * (wx - vx);
  const projY = vy + t * (wy - vy);

  return Math.sqrt((px - projX) * (px - projX) + (py - projY) * (py - projY));
}
