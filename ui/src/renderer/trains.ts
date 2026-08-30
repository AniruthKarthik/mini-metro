import type { TrainDTO, LineDTO, StationDTO, Pos, StationKind } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { getLineColor, generateOctilinearPath, buildSharedEdgeMap, getSegmentParallelOffset } from './lines';
import type { SharedEdgeMap } from './lines';
import { drawPassengerShape, DARK_CHARCOAL, WHITE_FILL } from './shapes';

export class TrainInterpolator {
  private previousById: Map<number, TrainDTO> = new Map();
  private currentById: Map<number, TrainDTO> = new Map();
  private previousAt = 0;
  private currentAt = 0;

  public setSnapshot(trains: TrainDTO[], now: number = performance.now()): void {
    const incoming = new Map((trains || []).map((train) => [train.id, train]));

    // BUG-5 fix: detect recycled train IDs. The engine reuses a deactivated train's
    // slot index (and thus its ID) when spawning a new train. If the previous snapshot
    // had train ID=N on lineA, and the new snapshot has train ID=N on lineB, carrying
    // over the previous interpolation state would cause a one-frame snap/teleport.
    // Clear the previous entry for any ID whose line has changed so interpolation
    // starts fresh for the re-activated train.
    const previous = this.currentById;
    for (const [id, train] of incoming) {
      const prev = previous.get(id);
      if (prev && prev.line_id !== train.line_id) {
        // Recycled slot: discard history so the new train starts without stale state.
        previous.delete(id);
      }
    }

    this.previousById = previous;
    this.previousAt = this.currentAt || now;
    this.currentById = incoming;
    this.currentAt = now;
  }

  public renderTrains(
    ctx: CanvasRenderingContext2D,
    viewport: Viewport,
    trains: TrainDTO[],
    lines: LineDTO[],
    stations: StationDTO[]
  ): void {
    ctx.save();

    const stationMap = new Map<number, StationDTO>();
    for (const st of stations) {
      stationMap.set(st.id, st);
    }

    const lineMap = new Map<number, LineDTO>();
    for (const l of lines) {
      lineMap.set(l.id, l);
    }

    const sharedEdgeMap = buildSharedEdgeMap(lines);

    const elapsed = this.currentAt > this.previousAt ? this.currentAt - this.previousAt : 0;
    const alpha = elapsed > 0 ? Math.max(0, Math.min(1, (performance.now() - this.currentAt) / elapsed)) : 1;

    for (const rawTrain of trains) {
      const tr = this.interpolateTrain(rawTrain, alpha);
      const line = lineMap.get(tr.line_id);
      if (!line || line.removed || !line.stations || line.stations.length < 2) {
        continue;
      }

      const trainPos = computeTrainPosition(tr, line, stationMap, viewport, sharedEdgeMap);
      if (!trainPos) continue;

      const color = getLineColor(tr.line_id);
      const allPassengers = tr.passengers || [];

      // Locomotive takes up to 6 passengers; attached carriages take remainder
      const locoPassengers = allPassengers.slice(0, 6);
      renderTrainCar(ctx, trainPos.pos, trainPos.angle, color, locoPassengers);

      if (tr.carriages > 1) {
        const carriageCap = 6;
        for (let c = 1; c < tr.carriages; c++) {
          const trailDist = c * 26;
          const trainX = getX(trainPos.pos);
          const trainY = getY(trainPos.pos);
          const trailPos = {
            x: trainX - Math.cos(trainPos.angle) * trailDist,
            y: trainY - Math.sin(trainPos.angle) * trailDist,
          };
          const startIdx = 6 + (c - 1) * carriageCap;
          const carriagePassengers = allPassengers.slice(startIdx, startIdx + carriageCap);
          renderCarriageCar(ctx, trailPos, trainPos.angle, color, carriagePassengers);
        }
      }
    }

    ctx.restore();
  }

  private interpolateTrain(current: TrainDTO, alpha: number): TrainDTO {
    const previous = this.previousById.get(current.id);
    if (
      !previous ||
      previous.line_id !== current.line_id ||
      previous.segment !== current.segment ||
      // BUG-15 (design): when direction flips at a terminal bounce, interpolation is
      // aborted and the train snaps to the current frame position. This is intentional:
      // interpolating through a direction reversal would show the train running backwards
      // through the terminal, which is visually worse than a single-frame snap.
      previous.direction !== current.direction ||
      Math.abs(current.progress - previous.progress) > 0.5
    ) {
      return current;
    }

    return {
      ...current,
      progress: previous.progress + (current.progress - previous.progress) * alpha,
    };
  }
}

export function computeTrainPosition(
  tr: TrainDTO,
  line: LineDTO,
  stationMap: Map<number, StationDTO>,
  viewport: Viewport,
  edgeMap: SharedEdgeMap
): { pos: Pos; angle: number } | null {
  const n = line.stations.length;
  if (n < 2 || tr.segment < 0 || tr.segment >= n) return null;

  // Segment endpoints in line station array order (0 to n-2)
  let segIdx = tr.segment;
  if (segIdx >= n - 1) {
    segIdx = line.is_loop ? (n - 1) : (n - 2);
  }

  let nextSegIdx = segIdx + 1;
  if (line.is_loop && nextSegIdx >= n) {
    nextSegIdx = 0;
  }

  const st1Id = line.stations[segIdx];
  const st2Id = line.stations[nextSegIdx];

  const st1 = stationMap.get(st1Id);
  const st2 = stationMap.get(st2Id);
  if (!st1 || !st2) return null;

  const p1 = viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
  const p2 = viewport.mapToScreen({ x: getX(st2), y: getY(st2) });

  // EXACT same parallel offset and octilinear path as lines.ts
  const { p1Offset, p2Offset } = getSegmentParallelOffset(p1, p2, st1Id, st2Id, tr.line_id, edgeMap, 8.0);
  const octilinearPts = generateOctilinearPath([p1Offset, p2Offset]);

  if (octilinearPts.length < 2) {
    return { pos: p1Offset, angle: 0 };
  }

  // Calculate segment lengths
  const segLengths: number[] = [];
  let totalLength = 0;

  for (let i = 0; i < octilinearPts.length - 1; i++) {
    const a = octilinearPts[i];
    const b = octilinearPts[i + 1];
    const dx = getX(b) - getX(a);
    const dy = getY(b) - getY(a);
    const len = Math.sqrt(dx * dx + dy * dy);
    segLengths.push(len);
    totalLength += len;
  }

  if (totalLength === 0) {
    return { pos: p1Offset, angle: 0 };
  }

  // Determine direction along the station array (forward or backward)
  const isMovingForward = tr.direction >= 0;
  const rawProg = Math.max(0, Math.min(1, tr.progress));
  const effectiveProg = isMovingForward ? rawProg : (1.0 - rawProg);
  let targetDist = effectiveProg * totalLength;

  // Interpolate along octilinearPts
  for (let i = 0; i < octilinearPts.length - 1; i++) {
    const len = segLengths[i];
    const a = octilinearPts[i];
    const b = octilinearPts[i + 1];

    const ax = getX(a), ay = getY(a);
    const bx = getX(b), by = getY(b);

    if (targetDist <= len || i === octilinearPts.length - 2) {
      const frac = len > 0 ? Math.max(0, Math.min(1, targetDist / len)) : 0;
      const x = ax + (bx - ax) * frac;
      const y = ay + (by - ay) * frac;

      let angle = Math.atan2(by - ay, bx - ax);
      if (!isMovingForward) {
        angle += Math.PI;
      }

      return { pos: { x, y }, angle };
    }
    targetDist -= len;
  }

  return { pos: p1Offset, angle: 0 };
}

function renderTrainCar(
  ctx: CanvasRenderingContext2D,
  pos: Pos,
  angle: number,
  color: string,
  passengers: StationKind[]
): void {
  const width = 28;
  const height = 17;
  const px = getX(pos), py = getY(pos);

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(angle);

  // Train shadow
  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.beginPath();
  ctx.roundRect(-width / 2 + 2, -height / 2 + 2, width, height, 4);
  ctx.fill();

  // Train body
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(-width / 2, -height / 2, width, height, 4);
  ctx.fill();

  ctx.strokeStyle = DARK_CHARCOAL;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Render all onboard passenger shapes (up to max capacity with 2-row layout if > 4)
  const count = passengers ? passengers.length : 0;
  if (count > 0) {
    if (count <= 4) {
      const startX = -width * 0.32;
      const spacing = (width * 0.64) / Math.max(1, count - 1);
      for (let i = 0; i < count; i++) {
        const dotX = count === 1 ? 0 : startX + i * spacing;
        drawPassengerShape(ctx, passengers[i], dotX, 0, 3.0, WHITE_FILL);
      }
    } else {
      const topCount = Math.min(4, Math.ceil(count / 2));
      const bottomCount = count - topCount;

      // Top row
      const startXTop = -width * 0.3;
      const spacingTop = (width * 0.6) / Math.max(1, topCount - 1);
      for (let i = 0; i < topCount; i++) {
        const dotX = topCount === 1 ? 0 : startXTop + i * spacingTop;
        drawPassengerShape(ctx, passengers[i], dotX, -3.6, 2.3, WHITE_FILL);
      }

      // Bottom row
      const startXBot = -width * 0.3;
      const spacingBot = (width * 0.6) / Math.max(1, bottomCount - 1);
      for (let j = 0; j < bottomCount; j++) {
        const dotX = bottomCount === 1 ? 0 : startXBot + j * spacingBot;
        drawPassengerShape(ctx, passengers[topCount + j], dotX, 3.6, 2.3, WHITE_FILL);
      }
    }
  }

  ctx.restore();
}

function renderCarriageCar(
  ctx: CanvasRenderingContext2D,
  pos: Pos,
  angle: number,
  color: string,
  passengers: StationKind[]
): void {
  const width = 22;
  const height = 15;
  const px = getX(pos), py = getY(pos);

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(angle);

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(-width / 2, -height / 2, width, height, 3);
  ctx.fill();

  ctx.strokeStyle = DARK_CHARCOAL;
  ctx.lineWidth = 1.8;
  ctx.stroke();

  const count = passengers ? passengers.length : 0;
  if (count > 0) {
    if (count <= 4) {
      const startX = -width * 0.3;
      const spacing = (width * 0.6) / Math.max(1, count - 1);
      for (let i = 0; i < count; i++) {
        const dotX = count === 1 ? 0 : startX + i * spacing;
        drawPassengerShape(ctx, passengers[i], dotX, 0, 2.6, WHITE_FILL);
      }
    } else {
      const topCount = Math.min(4, Math.ceil(count / 2));
      const bottomCount = count - topCount;
      const startXTop = -width * 0.28;
      const spacingTop = (width * 0.56) / Math.max(1, topCount - 1);
      for (let i = 0; i < topCount; i++) {
        const dotX = topCount === 1 ? 0 : startXTop + i * spacingTop;
        drawPassengerShape(ctx, passengers[i], dotX, -3.2, 2.1, WHITE_FILL);
      }
      const startXBot = -width * 0.28;
      const spacingBot = (width * 0.56) / Math.max(1, bottomCount - 1);
      for (let j = 0; j < bottomCount; j++) {
        const dotX = bottomCount === 1 ? 0 : startXBot + j * spacingBot;
        drawPassengerShape(ctx, passengers[topCount + j], dotX, 3.2, 2.1, WHITE_FILL);
      }
    }
  }

  ctx.restore();
}
