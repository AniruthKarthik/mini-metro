import type { TrainDTO, LineDTO, StationDTO, Pos } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { getLineColor } from './lines';
import { DARK_CHARCOAL, WHITE_FILL } from './shapes';

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

    for (const tr of trains) {
      const line = lineMap.get(tr.line_id);
      if (!line || line.removed || !line.stations || line.stations.length < 2) {
        continue;
      }

      const trainPos = computeTrainPosition(tr, line, stationMap, viewport);
      if (!trainPos) continue;

      const color = getLineColor(tr.line_id);

      renderTrainCar(ctx, trainPos.pos, trainPos.angle, color, tr.load);

      if (tr.carriages > 1) {
        for (let c = 1; c < tr.carriages; c++) {
          const trailDist = c * 22;
          const trainX = getX(trainPos.pos);
          const trainY = getY(trainPos.pos);
          const trailPos = {
            x: trainX - Math.cos(trainPos.angle) * trailDist,
            y: trainY - Math.sin(trainPos.angle) * trailDist,
          };
          renderCarriageCar(ctx, trailPos, trainPos.angle, color);
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
  viewport: Viewport
): { pos: Pos; angle: number } | null {
  const n = line.stations.length;
  if (tr.segment < 0 || tr.segment >= n) return null;

  let st1Id = line.stations[tr.segment];
  let nextIdx = tr.segment + tr.direction;

  if (line.is_loop) {
    nextIdx = (tr.segment + tr.direction + n) % n;
  } else {
    if (nextIdx < 0) nextIdx = 0;
    if (nextIdx >= n) nextIdx = n - 1;
  }

  let st2Id = line.stations[nextIdx];

  const st1 = stationMap.get(st1Id);
  const st2 = stationMap.get(st2Id);
  if (!st1 || !st2) return null;

  const p1 = viewport.mapToScreen({ x: st1.x, y: st1.y });
  const p2 = viewport.mapToScreen({ x: st2.x, y: st2.y });

  const p1x = getX(p1), p1y = getY(p1);
  const p2x = getX(p2), p2y = getY(p2);

  const prog = Math.max(0, Math.min(1, tr.progress));

  const x = p1x + (p2x - p1x) * prog;
  const y = p1y + (p2y - p1y) * prog;

  const angle = Math.atan2(p2y - p1y, p2x - p1x);

  return { pos: { x, y }, angle };
}

function renderTrainCar(
  ctx: CanvasRenderingContext2D,
  pos: Pos,
  angle: number,
  color: string,
  loadCount: number
): void {
  const width = 24;
  const height = 14;
  const px = getX(pos), py = getY(pos);

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(angle);

  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.beginPath();
  ctx.roundRect(-width / 2 + 2, -height / 2 + 2, width, height, 4);
  ctx.fill();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(-width / 2, -height / 2, width, height, 4);
  ctx.fill();

  ctx.strokeStyle = DARK_CHARCOAL;
  ctx.lineWidth = 2;
  ctx.stroke();

  if (loadCount > 0) {
    const dots = Math.min(4, loadCount);
    ctx.fillStyle = WHITE_FILL;
    for (let i = 0; i < dots; i++) {
      const dotX = -width / 4 + (i * width) / 5;
      ctx.beginPath();
      ctx.arc(dotX, 0, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.restore();
}

function renderCarriageCar(
  ctx: CanvasRenderingContext2D,
  pos: Pos,
  angle: number,
  color: string
): void {
  const width = 16;
  const height = 12;
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

  ctx.restore();
}
