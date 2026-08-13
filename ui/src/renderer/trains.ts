import type { TrainDTO, LineDTO, StationDTO, Pos, StationKind } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { getLineColor, generateOctilinearPath, buildSharedEdgeMap, getSegmentParallelOffset } from './lines';
import type { SharedEdgeMap } from './lines';
import { drawPassengerShape, DARK_CHARCOAL, WHITE_FILL } from './shapes';

export class TrainInterpolator {
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

    for (const tr of trains) {
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
        const carriageCap = 4;
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
}

function computeTrainPosition(
  tr: TrainDTO,
  line: LineDTO,
  stationMap: Map<number, StationDTO>,
  viewport: Viewport,
  edgeMap: SharedEdgeMap
): { pos: Pos; angle: number } | null {
  const n = line.stations.length;
  if (tr.segment < 0 || tr.segment >= n) return null;

  let st1Idx = tr.segment;
  let st2Idx = tr.segment + tr.direction;

  if (line.is_loop) {
    st2Idx = (tr.segment + tr.direction + n) % n;
  } else {
    if (st2Idx < 0) st2Idx = 0;
    if (st2Idx >= n) st2Idx = n - 1;
  }

  const st1Id = line.stations[st1Idx];
  const st2Id = line.stations[st2Idx];

  const st1 = stationMap.get(st1Id);
  const st2 = stationMap.get(st2Id);
  if (!st1 || !st2) return null;

  const p1 = viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
  const p2 = viewport.mapToScreen({ x: getX(st2), y: getY(st2) });

  const { p1Offset, p2Offset } = getSegmentParallelOffset(p1, p2, st1Id, st2Id, tr.line_id, edgeMap, 8.0);

  // Order endpoints canonically to match exact parallel track line geometry drawn by lines.ts
  const isForward = st1Idx <= st2Idx;
  const startP = isForward ? p1Offset : p2Offset;
  const endP = isForward ? p2Offset : p1Offset;

  // Generate the EXACT canonical 45° octilinear track path
  const octilinearPts = generateOctilinearPath([startP, endP]);
  if (octilinearPts.length < 2) {
    return { pos: p1Offset, angle: 0 };
  }

  // Calculate segment lengths along canonical octilinear path
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

  const rawProg = Math.max(0, Math.min(1, tr.progress));
  const effectiveProg = isForward ? rawProg : 1.0 - rawProg;
  let targetDist = effectiveProg * totalLength;

  // Interpolate position and angle along canonical octilinear path
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
      if (!isForward) {
        angle += Math.PI; // Reverse train orientation when moving backward along canonical track
      }

      return { pos: { x, y }, angle };
    }

    targetDist -= len;
  }

  return { pos: isForward ? p2Offset : p1Offset, angle: 0 };
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
