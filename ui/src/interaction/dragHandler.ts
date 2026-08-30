import type { StationDTO, LineDTO, TrainDTO, Pos, DragState } from '../types';
import { getX, getY } from '../types';
import { Viewport } from '../renderer/viewport';
import { GameWSClient } from '../ws/client';
import { getLineColor, generateOctilinearPath, getTerminalCapPosition, buildSharedEdgeMap, getSegmentParallelOffset } from '../renderer/lines';
import { computeTrainPosition } from '../renderer/trains';

const NEW_LINE_GUIDE_COLOR = '#888888';

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

    // Right-click to remove line & refund resources
    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const pos = this.getEventPos(e);
      const snap = this.wsClient.getSnapshot();
      if (!snap) return;

      const lines = snap.lines || [];
      const stations = snap.stations || [];

      // Right-click terminal handle -> Remove Line
      const terminalHit = this.findTerminalToExtend(pos, lines, stations);
      if (terminalHit) {
        console.log(`🗑️ [FRONTEND] Right-click removing Line ${terminalHit.lineId}`);
        this.wsClient.sendAction({
          type: 'remove_line',
          payload: { line_id: terminalHit.lineId },
        });
        return;
      }

      // Right-click track segment -> Remove Line
      const segHit = this.findSegmentToInsert(pos, lines, stations);
      if (segHit) {
        console.log(`🗑️ [FRONTEND] Right-click removing Line ${segHit.lineId}`);
        this.wsClient.sendAction({
          type: 'remove_line',
          payload: { line_id: segHit.lineId },
        });
        return;
      }
    });
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
        return { points: [p1, p2], color: NEW_LINE_GUIDE_COLOR };
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
      if ((this.dragState.source as any).lineId === -1) color = getLineColor(this.getNextLineIndex(lines));

      const fromSt = stations.find((s) => s.id === (this.dragState!.source as any).fromStationId);
      if (fromSt) {
        points.push(this.viewport.mapToScreen({ x: getX(fromSt), y: getY(fromSt) }));
      }
    } else if ((this.dragState.source as any).type === 'insert_station') {
      const src = this.dragState.source as any;
      const line = lines.find((l) => l.id === src.lineId);
      if (line) color = getLineColor(line.id);

      const stA = stations.find((s) => s.id === src.fromStationAId);
      const stB = stations.find((s) => s.id === src.toStationBId);

      const currentScreen = this.dragState.targetStationId !== null
        ? (() => {
            const st = stations.find((s) => s.id === this.dragState!.targetStationId);
            return st ? this.viewport.mapToScreen({ x: getX(st), y: getY(st) }) : this.dragState!.currentPos;
          })()
        : this.dragState.currentPos;

      if (stA && stB) {
        const pA = this.viewport.mapToScreen({ x: getX(stA), y: getY(stA) });
        const pB = this.viewport.mapToScreen({ x: getX(stB), y: getY(stB) });
        return { points: [pA, currentScreen, pB], color };
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

  public renderDragOverlay(ctx: CanvasRenderingContext2D): void {
    if (!this.dragState) return;

    const px = getX(this.dragState.currentPos);
    const py = getY(this.dragState.currentPos);
    const sourceType = (this.dragState.source as any).type;

    ctx.save();

    if (sourceType === 'add_train' || sourceType === 'reposition_train') {
      // Floating Locomotive Icon under cursor
      const width = 34;
      const height = 20;

      ctx.save();
      ctx.translate(px, py);

      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      ctx.beginPath();
      ctx.roundRect(-width / 2 + 3, -height / 2 + 4, width, height, 5);
      ctx.fill();

      ctx.fillStyle = '#22252a';
      ctx.beginPath();
      ctx.roundRect(-width / 2, -height / 2, width, height, 5);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(-8, 7, 2.5, 0, Math.PI * 2);
      ctx.arc(8, 7, 2.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    } else if (sourceType === 'add_carriage') {
      // Floating Carriage Icon under cursor
      const width = 26;
      const height = 16;

      ctx.save();
      ctx.translate(px, py);

      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.beginPath();
      ctx.roundRect(-width / 2 + 2, -height / 2 + 3, width, height, 4);
      ctx.fill();

      ctx.fillStyle = '#444850';
      ctx.beginPath();
      ctx.roundRect(-width / 2, -height / 2, width, height, 4);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.restore();
    } else if (sourceType === 'upgrade_interchange') {
      // Floating Interchange Hub Icon under cursor
      const r = 12;

      ctx.save();
      ctx.translate(px, py);

      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.beginPath();
      ctx.arc(2, 3, r + 4, 0, Math.PI * 2);
      ctx.fill();

      // Double capsule ring
      ctx.beginPath();
      ctx.arc(0, 0, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = '#22252a';
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#22252a';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.restore();
    } else {
      // Floating line drag token cursor indicator
      ctx.save();
      ctx.translate(px, py);

      const color = sourceType === 'new_line'
        ? getLineColor((this.dragState.source as any).lineIndex)
        : sourceType === 'extend_line'
        ? getLineColor((this.dragState.source as any).lineId)
        : '#22252a';

      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.beginPath();
      ctx.arc(2, 3, 11, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(0, 0, 10, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.restore();
    }

    ctx.restore();
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

  private findTrainAt(pos: Pos, trains: TrainDTO[], lines: LineDTO[], stations: StationDTO[], threshold: number = 45): { trainId: number; lineId: number } | null {
    if (!trains || !lines || !stations) return null;
    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) stationMap.set(st.id, st);

    const lineMap = new Map<number, LineDTO>();
    for (const l of lines) lineMap.set(l.id, l);

    const sharedEdgeMap = buildSharedEdgeMap(lines);
    const px = getX(pos), py = getY(pos);
    let closest: { trainId: number; lineId: number; dist: number } | null = null;

    for (const tr of trains) {
      if ((tr as any).active === false) continue;
      const line = lineMap.get(tr.line_id);
      if (!line || line.removed || !line.stations || line.stations.length < 2) continue;

      const trainPos = computeTrainPosition(tr, line, stationMap, this.viewport, sharedEdgeMap);
      if (!trainPos) continue;

      const dx = getX(trainPos.pos) - px;
      const dy = getY(trainPos.pos) - py;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist <= threshold) {
        if (!closest || dist < closest.dist) {
          closest = { trainId: tr.id, lineId: tr.line_id, dist };
        }
      }
    }

    return closest ? { trainId: closest.trainId, lineId: closest.lineId } : null;
  }

  private findTerminalToExtend(pos: Pos, lines: LineDTO[], stations: StationDTO[]): { lineId: number; stationId: number; fromFront: boolean } | null {
    if (!lines || !stations) return null;
    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) stationMap.set(st.id, st);

    const sharedEdgeMap = buildSharedEdgeMap(lines);

    const px = getX(pos), py = getY(pos);
    let bestMatch: { lineId: number; stationId: number; fromFront: boolean; dist: number } | null = null;

    for (const line of lines) {
      if (line.removed || !line.stations || line.stations.length < 2 || line.is_loop) continue;

      const firstStId = line.stations[0];
      const secondStId = line.stations[1];

      const lastStId = line.stations[line.stations.length - 1];
      const secondLastStId = line.stations[line.stations.length - 2];

      const firstSt = stationMap.get(firstStId);
      const secondSt = stationMap.get(secondStId);

      const lastSt = stationMap.get(lastStId);
      const secondLastSt = stationMap.get(secondLastStId);

      // 1. Back terminal end-cap T-bar & endpoint station
      if (lastSt) {
        const lastP = this.viewport.mapToScreen({ x: getX(lastSt), y: getY(lastSt) });
        const distSt = Math.sqrt((lastP.x - px) ** 2 + (lastP.y - py) ** 2);

        let distCap = 999;
        if (secondLastSt) {
          const prevP = this.viewport.mapToScreen({ x: getX(secondLastSt), y: getY(secondLastSt) });
          const { p1Offset, p2Offset } = getSegmentParallelOffset(lastP, prevP, lastStId, secondLastStId, line.id, sharedEdgeMap, 8.0);
          const capInfo = getTerminalCapPosition(p1Offset, p2Offset, 18);
          distCap = Math.sqrt((getX(capInfo.capPos) - px) ** 2 + (getY(capInfo.capPos) - py) ** 2);
        }

        const minDist = Math.min(distSt, distCap);
        if (minDist <= 55) {
          if (!bestMatch || minDist < bestMatch.dist) {
            bestMatch = { lineId: line.id, stationId: lastStId, fromFront: false, dist: minDist };
          }
        }
      }

      // 2. Front terminal end-cap T-bar & endpoint station
      if (firstSt) {
        const firstP = this.viewport.mapToScreen({ x: getX(firstSt), y: getY(firstSt) });
        const distSt = Math.sqrt((firstP.x - px) ** 2 + (firstP.y - py) ** 2);

        let distCap = 999;
        if (secondSt) {
          const nextP = this.viewport.mapToScreen({ x: getX(secondSt), y: getY(secondSt) });
          // BUG-16 fix: use the offset-adjusted second-station position (p2Offset) rather
          // than the raw nextP. Previously the cap direction was computed from a shifted
          // p1Offset to an unshifted nextP, producing a wrong direction vector.
          const { p1Offset, p2Offset } = getSegmentParallelOffset(firstP, nextP, firstStId, secondStId, line.id, sharedEdgeMap, 8.0);
          const capInfo = getTerminalCapPosition(p1Offset, p2Offset, 18);
          distCap = Math.sqrt((getX(capInfo.capPos) - px) ** 2 + (getY(capInfo.capPos) - py) ** 2);
        }

        const minDist = Math.min(distSt, distCap);
        if (minDist <= 55) {
          if (!bestMatch || minDist < bestMatch.dist) {
            bestMatch = { lineId: line.id, stationId: firstStId, fromFront: true, dist: minDist };
          }
        }
      }
    }

    if (bestMatch) {
      return { lineId: bestMatch.lineId, stationId: bestMatch.stationId, fromFront: bestMatch.fromFront };
    }

    return null;
  }

  private findSegmentToInsert(pos: Pos, lines: LineDTO[], stations: StationDTO[], threshold: number = 35): { lineId: number; insertIndex: number; fromStationAId: number; toStationBId: number } | null {
    if (!lines || !stations) return null;
    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) stationMap.set(st.id, st);

    const sharedEdgeMap = buildSharedEdgeMap(lines);

    const px = getX(pos);
    const py = getY(pos);
    let bestMatch: { lineId: number; insertIndex: number; fromStationAId: number; toStationBId: number; dist: number } | null = null;

    for (const line of lines) {
      if (line.removed || !line.stations || line.stations.length < 2) continue;

      for (let i = 0; i < line.stations.length - 1; i++) {
        const st1Id = line.stations[i];
        const st2Id = line.stations[i + 1];

        const st1 = stationMap.get(st1Id);
        const st2 = stationMap.get(st2Id);
        if (!st1 || !st2) continue;

        const p1 = this.viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
        const p2 = this.viewport.mapToScreen({ x: getX(st2), y: getY(st2) });

        // Generate exact parallel 45° octilinear path matching rendered track line geometry
        const { p1Offset, p2Offset } = getSegmentParallelOffset(p1, p2, st1Id, st2Id, line.id, sharedEdgeMap, 8.0);
        const octPts = generateOctilinearPath([p1Offset, p2Offset]);

        for (let k = 0; k < octPts.length - 1; k++) {
          const dist = distToSegment({ x: px, y: py }, octPts[k], octPts[k + 1]);
          if (dist <= threshold) {
            if (!bestMatch || dist < bestMatch.dist) {
              bestMatch = {
                lineId: line.id,
                insertIndex: i + 1,
                fromStationAId: st1Id,
                toStationBId: st2Id,
                dist,
              };
            }
          }
        }
      }
    }

    return bestMatch
      ? {
          lineId: bestMatch.lineId,
          insertIndex: bestMatch.insertIndex,
          fromStationAId: bestMatch.fromStationAId,
          toStationBId: bestMatch.toStationBId,
        }
      : null;
  }



  private getNextLineIndex(lines: LineDTO[]): number {
    const activeIds = new Set(lines.filter((line) => !line.removed).map((line) => line.id));
    for (let i = 0; i < lines.length + 8; i++) {
      if (!activeIds.has(i)) return i;
    }
    return lines.length;
  }

  public startDragNewLine(lineIndex: number, startPos: Pos): void {
    const snap = this.wsClient.getSnapshot();
    const stations = snap ? snap.stations || [] : [];
    const st = this.findStationAt(startPos, stations, 120);

    this.dragState = {
      source: { type: 'new_line', lineIndex, firstStationId: st ? st.id : null } as any,
      currentPos: startPos,
      targetStationId: null,
    };
  }

  public startDragTrain(startPos: Pos): void {
    this.dragState = {
      source: { type: 'add_train' },
      currentPos: startPos,
      targetStationId: null,
    };
  }

  public startDragCarriage(startPos: Pos): void {
    this.dragState = {
      source: { type: 'add_carriage' },
      currentPos: startPos,
      targetStationId: null,
    };
  }

  public startDragInterchange(startPos: Pos): void {
    this.dragState = {
      source: { type: 'upgrade_interchange' } as any,
      currentPos: startPos,
      targetStationId: null,
    };
  }

  private onPointerDown(e: MouseEvent): void {
    if (e.button !== 0) return; // Left click only

    const pos = this.getEventPos(e);
    const snap = this.wsClient.getSnapshot();
    if (!snap) return;

    const stations = snap.stations || [];
    const lines = snap.lines || [];
    const trains = snap.trains || [];
    const res = (snap.resources || {}) as any;
    const availableLines = res.lines ?? res.Lines ?? 0;

    // 0. Check active train hit box -> REPOSITION TRAIN
    const trainHit = this.findTrainAt(pos, trains, lines, stations, 45);
    if (trainHit) {
      this.dragState = {
        source: { type: 'reposition_train', trainId: trainHit.trainId, fromLineId: trainHit.lineId } as any,
        currentPos: pos,
        targetStationId: null,
      };
      this.selectedStationId = null;
      return;
    }

    // 1. Check line terminal extension hit box (extending T-bar handles or terminal station)
    const terminalHit = this.findTerminalToExtend(pos, lines, stations);
    if (terminalHit) {
      this.dragState = {
        source: { type: 'extend_line', lineId: terminalHit.lineId, fromStationId: terminalHit.stationId, fromFront: terminalHit.fromFront },
        currentPos: pos,
        targetStationId: null,
      };
      this.selectedStationId = null;
      return;
    }

    // 2. Check station node hit box -> DRAFT A NEW LINE FROM THIS STATION
    const st = this.findStationAt(pos, stations, 55);
    if (st) {
      if (availableLines > 0) {
        const nextLineIdx = this.getNextLineIndex(lines);
        this.dragState = {
          source: { type: 'new_line', lineIndex: nextLineIdx, firstStationId: st.id } as any,
          currentPos: pos,
          targetStationId: null,
        };
      } else {
        console.warn('⚠️ [FRONTEND] No available line tokens in resource pool');
      }
      this.selectedStationId = null;
      return;
    }

    // 3. Check line track segment hit box -> INSERT INTERMEDIATE STATION
    const segHit = this.findSegmentToInsert(pos, lines, stations);
    if (segHit) {
      this.dragState = {
        source: {
          type: 'insert_station',
          lineId: segHit.lineId,
          insertIndex: segHit.insertIndex,
          fromStationAId: segHit.fromStationAId,
          toStationBId: segHit.toStationBId,
        } as any,
        currentPos: pos,
        targetStationId: null,
      };
      this.selectedStationId = null;
      return;
    }

    this.selectedStationId = null;
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

      if (this.dragState.source.type === 'new_line') {
        const src = this.dragState.source as any;
        // BUG-17 fix: only lock firstStationId when the user actively hovers over a
        // *different* station while the drag is already in progress. This prevents the
        // very first station the cursor brushes during a HUD-initiated drag from being
        // permanently locked as the line origin before the user reaches their intended
        // target station.
        if (st && src.firstStationId === null) {
          // Only snap firstStationId if we have genuinely hovered over a station with
          // intent (i.e. the cursor is actually close enough to the station centre,
          // which findStationAt already enforces via its threshold).
          src.firstStationId = st.id;
        } else if (st && src.firstStationId !== null && st.id !== src.firstStationId) {
          this.dragState.targetStationId = st.id;
        } else if (!st) {
          this.dragState.targetStationId = null;
        }
      } else {
        this.dragState.targetStationId = this.hoveredStationId;
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

    if (snap) {
      if ((source as any).type === 'insert_station') {
        const src = source as any;
        if (targetStId !== null && targetStId !== src.fromStationAId && targetStId !== src.toStationBId) {
          console.log(`📤 [FRONTEND] Dispatching insert_station: Line ${src.lineId} Station ${targetStId} at index ${src.insertIndex}`);
          this.wsClient.sendAction({
            type: 'insert_station',
            payload: { line_id: src.lineId, station_id: targetStId, index: src.insertIndex, use_tunnel: false },
          });
          this.selectedStationId = null;
        }
      } else if (source.type === 'new_line') {
        const src = source as any;
        const firstStId = src.firstStationId;
        const secondStId = targetStId;

        if (firstStId !== null && secondStId !== null && firstStId !== secondStId) {
          this.wsClient.sendAction({
            type: 'add_line',
            payload: { stations: [firstStId, secondStId] },
          });
          this.selectedStationId = null;
        }
      } else if (source.type === 'extend_line') {
        const line = lines.find((l) => l.id === source.lineId);
        if (line) {
          // 1. Dragged endpoint into empty space -> Remove Line
          if (targetStId === null) {
            console.log(`🗑️ [FRONTEND] Dragged end off into space -> Removing Line ${line.id}`);
            this.wsClient.sendAction({
              type: 'remove_line',
              payload: { line_id: line.id },
            });
            this.selectedStationId = null;
            this.dragState = null;
            return;
          }

          // 2. Connecting opposite terminal -> Close loop
          if (!line.is_loop && line.stations.length >= 2) {
            const firstStId = line.stations[0];
            const lastStId = line.stations[line.stations.length - 1];

            const isOppositeTerminal =
              (source.fromStationId === firstStId && targetStId === lastStId) ||
              (source.fromStationId === lastStId && targetStId === firstStId);

            if (isOppositeTerminal) {
              console.log(`🔄 [FRONTEND] Closing loop for line ${line.id}`);
              this.wsClient.sendAction({
                type: 'close_loop',
                payload: { line_id: line.id, use_tunnel: false },
              });
              this.selectedStationId = null;
              this.dragState = null;
              return;
            }

            // 3. Shortening 2-station line back onto origin -> Remove Line
            if (line.stations.length === 2 && (targetStId === firstStId || targetStId === lastStId)) {
              console.log(`🗑️ [FRONTEND] Shortening 2-station line to origin -> Removing Line ${line.id}`);
              this.wsClient.sendAction({
                type: 'remove_line',
                payload: { line_id: line.id },
              });
              this.selectedStationId = null;
              this.dragState = null;
              return;
            }
          }

          // 4. Normal line extension
          if (!line.stations.includes(targetStId) && source.fromStationId !== targetStId) {
            this.wsClient.sendAction({
              type: 'extend_line',
              payload: { line_id: source.lineId, station_id: targetStId, use_tunnel: false, from_front: source.fromFront },
            });
            this.selectedStationId = null;
          }
        }
      } else if (source.type === 'add_train') {
        const closestSeg = this.findSegmentToInsert(currentPos, lines, stations, 120);
        const targetLine = closestSeg
          ? lines.find((l) => l.id === closestSeg.lineId) || null
          : (targetStId !== null ? lines.find((l) => !l.removed && l.stations.includes(targetStId)) || null : null);

        if (targetLine) {
          console.log(`🚂 [FRONTEND] Adding locomotive to Line ${targetLine.id}`);
          this.wsClient.sendAction({
            type: 'add_train',
            payload: { line_id: targetLine.id },
          });
        }
      } else if (source.type === 'reposition_train' || (source as any).type === 'reposition_train') {
        const src = source as any;
        const closestSeg = this.findSegmentToInsert(currentPos, lines, stations, 120);
        let targetLine: LineDTO | null = null;
        let segIdx = 0;

        if (closestSeg) {
          targetLine = lines.find((l) => l.id === closestSeg.lineId) || null;
          segIdx = Math.max(0, closestSeg.insertIndex - 1);
        } else if (targetStId !== null) {
          targetLine = lines.find((l) => !l.removed && l.stations.includes(targetStId)) || null;
          if (targetLine) {
            const stIdx = targetLine.stations.indexOf(targetStId);
            segIdx = Math.max(0, Math.min(targetLine.stations.length - 1, stIdx));
          }
        }

        if (targetLine) {
          console.log(`🚂 [FRONTEND] Moving Train ${src.trainId} (from Line ${src.fromLineId}) to Line ${targetLine.id} at segment ${segIdx}`);
          this.wsClient.sendAction({
            type: 'reposition_train',
            payload: { train_id: src.trainId, line_id: targetLine.id, segment: segIdx, direction: 1 },
          });
        }
      } else if (source.type === 'add_carriage') {
        const closestSeg = this.findSegmentToInsert(currentPos, lines, stations, 120);
        const targetLine = closestSeg
          ? lines.find((l) => l.id === closestSeg.lineId) || null
          : (targetStId !== null ? lines.find((l) => !l.removed && l.stations.includes(targetStId)) || null : null);

        // BUG-18 fix: pick the *closest* active train on the target line by screen
        // distance using computeTrainPosition (already statically imported from trains.ts),
        // rather than always using the first train in the array.
        let train: typeof trains[0] | undefined;
        if (targetLine) {
          const stationMap = new Map(stations.map((s) => [s.id, s]));
          const sharedEdgeMap = buildSharedEdgeMap(lines);
          let minDist = Infinity;
          for (const tr of trains) {
            if (tr.line_id !== targetLine.id) continue;
            const trainPos = computeTrainPosition(tr, targetLine, stationMap, this.viewport, sharedEdgeMap);
            if (trainPos) {
              const dx = getX(trainPos.pos) - getX(currentPos);
              const dy = getY(trainPos.pos) - getY(currentPos);
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < minDist) {
                minDist = dist;
                train = tr;
              }
            }
          }
        }
        if (train) {
          console.log(`🚃 [FRONTEND] Adding carriage to Train ${train.id} on Line ${targetLine!.id}`);
          this.wsClient.sendAction({
            type: 'add_carriage',
            payload: { train_id: train.id },
          });
        }
      } else if ((source as any).type === 'upgrade_interchange') {
        if (targetStId !== null) {
          console.log(`🌟 [FRONTEND] Dispatching upgrade_interchange for Station ${targetStId}`);
          this.wsClient.sendAction({
            type: 'upgrade_interchange',
            payload: { station_id: targetStId },
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
