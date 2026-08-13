import type { LineDTO, StationDTO, Pos, RiverSegment } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { LINE_COLORS } from './shapes';

export function getLineColor(lineId: number): string {
  return LINE_COLORS[lineId % LINE_COLORS.length];
}

export type SharedEdgeKey = string;

export function getSharedEdgeKey(stA: number, stB: number): SharedEdgeKey {
  return stA < stB ? `${stA}-${stB}` : `${stB}-${stA}`;
}

export type SharedEdgeMap = Map<SharedEdgeKey, number[]>;

export function buildSharedEdgeMap(lines: LineDTO[]): SharedEdgeMap {
  const map: SharedEdgeMap = new Map();

  for (const line of lines) {
    if (line.removed || !line.stations || line.stations.length < 2) continue;

    for (let i = 0; i < line.stations.length - 1; i++) {
      const st1 = line.stations[i];
      const st2 = line.stations[i + 1];
      const key = getSharedEdgeKey(st1, st2);

      if (!map.has(key)) {
        map.set(key, []);
      }
      const lineList = map.get(key)!;
      if (!lineList.includes(line.id)) {
        lineList.push(line.id);
      }
    }

    if (line.is_loop && line.stations.length >= 2) {
      const st1 = line.stations[line.stations.length - 1];
      const st2 = line.stations[0];
      const key = getSharedEdgeKey(st1, st2);

      if (!map.has(key)) {
        map.set(key, []);
      }
      const lineList = map.get(key)!;
      if (!lineList.includes(line.id)) {
        lineList.push(line.id);
      }
    }
  }

  return map;
}

export function getSegmentParallelOffset(
  p1: Pos,
  p2: Pos,
  st1Id: number,
  st2Id: number,
  lineId: number,
  sharedEdgeMap: SharedEdgeMap,
  trackLineWidth: number = 8.0
): { p1Offset: Pos; p2Offset: Pos } {
  const key = getSharedEdgeKey(st1Id, st2Id);
  const sharingLines = sharedEdgeMap.get(key) || [lineId];

  const totalLinesOnSegment = sharingLines.length;
  if (totalLinesOnSegment <= 1) {
    return { p1Offset: p1, p2Offset: p2 };
  }

  const sortedLines = [...sharingLines].sort((a, b) => a - b);
  const indexOnSegment = sortedLines.indexOf(lineId);
  if (indexOnSegment === -1) {
    return { p1Offset: p1, p2Offset: p2 };
  }

  const dx = getX(p2) - getX(p1);
  const dy = getY(p2) - getY(p1);
  const len = Math.sqrt(dx * dx + dy * dy);

  if (len === 0) {
    return { p1Offset: p1, p2Offset: p2 };
  }

  const nx = -dy / len;
  const ny = dx / len;

  const shiftMultiplier = indexOnSegment - (totalLinesOnSegment - 1) / 2;
  const shiftAmount = shiftMultiplier * trackLineWidth;

  return {
    p1Offset: { x: getX(p1) + nx * shiftAmount, y: getY(p1) + ny * shiftAmount },
    p2Offset: { x: getX(p2) + nx * shiftAmount, y: getY(p2) + ny * shiftAmount },
  };
}

export function generateOctilinearPath(points: Pos[]): Pos[] {
  if (points.length < 2) return points;

  const result: Pos[] = [points[0]];

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];

    const x1 = getX(p1), y1 = getY(p1);
    const x2 = getX(p2), y2 = getY(p2);

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);

    if (dx < 4 || dy < 4 || Math.abs(dx - dy) < 4) {
      result.push(p2);
      continue;
    }

    let mx: number, my: number;
    const signX = x2 > x1 ? 1 : -1;
    const signY = y2 > y1 ? 1 : -1;

    if (dx > dy) {
      mx = x1 + signX * dy;
      my = y2;
    } else {
      mx = x2;
      my = y1 + signY * dx;
    }

    result.push({ x: mx, y: my });
    result.push(p2);
  }

  return result;
}

export function renderMetroLines(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  lines: LineDTO[],
  stations: StationDTO[],
  activeDragLinePreview?: { points: Pos[]; color: string },
  rivers?: RiverSegment[]
): void {
  ctx.save();

  const stationMap = new Map<number, StationDTO>();
  for (const st of stations) {
    stationMap.set(st.id, st);
  }

  const sharedEdgeMap = buildSharedEdgeMap(lines);

  for (const line of lines) {
    if (line.removed || !line.stations || line.stations.length < 2) {
      continue;
    }

    const color = getLineColor(line.id);
    const numSegs = line.stations.length - 1;

    for (let i = 0; i < numSegs; i++) {
      const st1Id = line.stations[i];
      const st2Id = line.stations[i + 1];
      const st1 = stationMap.get(st1Id);
      const st2 = stationMap.get(st2Id);
      if (!st1 || !st2) continue;

      const p1 = viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
      const p2 = viewport.mapToScreen({ x: getX(st2), y: getY(st2) });

      const { p1Offset, p2Offset } = getSegmentParallelOffset(p1, p2, st1Id, st2Id, line.id, sharedEdgeMap, 8.0);
      const octSeg = generateOctilinearPath([p1Offset, p2Offset]);

      drawOctilinearTrack(ctx, octSeg, color, 8.0);

      // Tunnel / Bridge marker at exact river crossing intersection
      if (line.tunnel_at && line.tunnel_at[i]) {
        const riverHit = findRiverIntersection(st1, st2, rivers || []);
        const markerScreenPos = riverHit
          ? viewport.mapToScreen(riverHit)
          : { x: (getX(p1Offset) + getX(p2Offset)) / 2, y: (getY(p1Offset) + getY(p2Offset)) / 2 };

        drawBridgeMarkerAt(ctx, markerScreenPos, p1Offset, p2Offset);
      }
    }

    // Closed loop wrap-around segment
    if (line.is_loop && line.stations.length >= 2) {
      const stFirstId = line.stations[0];
      const stLastId = line.stations[line.stations.length - 1];
      const stFirst = stationMap.get(stFirstId);
      const stLast = stationMap.get(stLastId);
      if (stFirst && stLast) {
        const pLast = viewport.mapToScreen({ x: getX(stLast), y: getY(stLast) });
        const pFirst = viewport.mapToScreen({ x: getX(stFirst), y: getY(stFirst) });

        const { p1Offset, p2Offset } = getSegmentParallelOffset(pLast, pFirst, stLastId, stFirstId, line.id, sharedEdgeMap, 8.0);
        const octSeg = generateOctilinearPath([p1Offset, p2Offset]);
        drawOctilinearTrack(ctx, octSeg, color, 8.0);

        if (line.loop_tunnel) {
          const riverHit = findRiverIntersection(stLast, stFirst, rivers || []);
          const markerScreenPos = riverHit
            ? viewport.mapToScreen(riverHit)
            : { x: (getX(p1Offset) + getX(p2Offset)) / 2, y: (getY(p1Offset) + getY(p2Offset)) / 2 };

          drawBridgeMarkerAt(ctx, markerScreenPos, p1Offset, p2Offset);
        }
      }
    }

    // Terminal extending T-bar handle caps
    if (!line.is_loop && line.stations.length >= 2) {
      const st0 = stationMap.get(line.stations[0]);
      const st1 = stationMap.get(line.stations[1]);
      if (st0 && st1) {
        const p0 = viewport.mapToScreen({ x: getX(st0), y: getY(st0) });
        const p1 = viewport.mapToScreen({ x: getX(st1), y: getY(st1) });
        const { p1Offset, p2Offset } = getSegmentParallelOffset(p0, p1, line.stations[0], line.stations[1], line.id, sharedEdgeMap, 8.0);
        drawTerminalEndCap(ctx, p1Offset, p2Offset, color, 8.0);
      }

      const lastIdx = line.stations.length - 1;
      const stEnd = stationMap.get(line.stations[lastIdx]);
      const stPrev = stationMap.get(line.stations[lastIdx - 1]);
      if (stEnd && stPrev) {
        const pEnd = viewport.mapToScreen({ x: getX(stEnd), y: getY(stEnd) });
        const pPrev = viewport.mapToScreen({ x: getX(stPrev), y: getY(stPrev) });
        const { p1Offset, p2Offset } = getSegmentParallelOffset(pEnd, pPrev, line.stations[lastIdx], line.stations[lastIdx - 1], line.id, sharedEdgeMap, 8.0);
        drawTerminalEndCap(ctx, p1Offset, p2Offset, color, 8.0);
      }
    }
  }

  // Draw active drag preview with octilinear routing
  if (activeDragLinePreview && activeDragLinePreview.points.length >= 2) {
    const octilinearPreview = generateOctilinearPath(activeDragLinePreview.points);
    drawOctilinearTrack(ctx, octilinearPreview, activeDragLinePreview.color, 7.5, true);
  }

  ctx.restore();
}

export function drawOctilinearTrack(
  ctx: CanvasRenderingContext2D,
  points: Pos[],
  color: string,
  lineWidth: number = 8.0,
  isDashed: boolean = false
): void {
  if (points.length < 2) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(getX(points[0]), getY(points[0]));

  const filletRadius = 14;

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];

    const p1x = getX(p1), p1y = getY(p1);
    const p2x = getX(p2), p2y = getY(p2);

    if (i < points.length - 2) {
      const p3 = points[i + 2];
      const p3x = getX(p3), p3y = getY(p3);

      const d1x = p2x - p1x, d1y = p2y - p1y;
      const len1 = Math.sqrt(d1x * d1x + d1y * d1y);
      const d2x = p3x - p2x, d2y = p3y - p2y;
      const len2 = Math.sqrt(d2x * d2x + d2y * d2y);

      if (len1 > 0 && len2 > 0) {
        const cut1 = Math.min(filletRadius, len1 / 2);
        const cut2 = Math.min(filletRadius, len2 / 2);

        const startX = p2x - (d1x / len1) * cut1;
        const startY = p2y - (d1y / len1) * cut1;
        const endX = p2x + (d2x / len2) * cut2;
        const endY = p2y + (d2y / len2) * cut2;

        ctx.lineTo(startX, startY);
        ctx.quadraticCurveTo(p2x, p2y, endX, endY);
      } else {
        ctx.lineTo(p2x, p2y);
      }
    } else {
      ctx.lineTo(p2x, p2y);
    }
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (isDashed) {
    ctx.setLineDash([10, 6]);
  }

  ctx.stroke();
  ctx.restore();
}

export function getTerminalCapPosition(
  terminal: Pos,
  nextPt: Pos,
  capOffsetLength: number = 18
): { capPos: Pos; nx: number; ny: number } {
  const tx = getX(terminal), ty = getY(terminal);
  const nx = getX(nextPt), ny = getY(nextPt);

  const dx = tx - nx;
  const dy = ty - ny;
  const len = Math.sqrt(dx * dx + dy * dy);

  if (len === 0) {
    return { capPos: terminal, nx: 0, ny: 1 };
  }

  const ux = dx / len;
  const uy = dy / len;

  const capX = tx + ux * capOffsetLength;
  const capY = ty + uy * capOffsetLength;

  const perpX = -uy;
  const perpY = ux;

  return {
    capPos: { x: capX, y: capY },
    nx: perpX,
    ny: perpY,
  };
}

export function drawTerminalEndCap(
  ctx: CanvasRenderingContext2D,
  terminal: Pos,
  nextPt: Pos,
  color: string,
  lineWidth: number
): void {
  const { capPos, nx, ny } = getTerminalCapPosition(terminal, nextPt, 18);
  const termX = getX(terminal), termY = getY(terminal);
  const capX = getX(capPos), capY = getY(capPos);

  const capWidth = lineWidth * 1.85;

  ctx.save();

  ctx.beginPath();
  ctx.moveTo(termX, termY);
  ctx.lineTo(capX, capY);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth * 0.85;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(capX - nx * capWidth, capY - ny * capWidth);
  ctx.lineTo(capX + nx * capWidth, capY + ny * capWidth);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth * 1.3;
  ctx.lineCap = 'round';
  ctx.stroke();

  ctx.strokeStyle = 'rgba(34, 37, 42, 0.45)';
  ctx.lineWidth = 1.8;
  ctx.stroke();

  ctx.restore();
}

function drawBridgeMarkerAt(ctx: CanvasRenderingContext2D, centerPos: Pos, p1: Pos, p2: Pos): void {
  const cx = getX(centerPos), cy = getY(centerPos);
  const p1x = getX(p1), p1y = getY(p1);
  const p2x = getX(p2), p2y = getY(p2);

  const dx = p2x - p1x;
  const dy = p2y - p1y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return;

  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;

  const bridgeW = 11;
  const gap = 4.5;

  ctx.save();
  ctx.strokeStyle = '#22252a';
  ctx.lineWidth = 3.0;

  ctx.beginPath();
  ctx.moveTo(cx - ux * gap - nx * bridgeW, cy - uy * gap - ny * bridgeW);
  ctx.lineTo(cx - ux * gap + nx * bridgeW, cy - uy * gap + ny * bridgeW);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx + ux * gap - nx * bridgeW, cy + uy * gap - ny * bridgeW);
  ctx.lineTo(cx + ux * gap + nx * bridgeW, cy + uy * gap + ny * bridgeW);
  ctx.stroke();

  ctx.restore();
}

export function findRiverIntersection(
  st1: Pos,
  st2: Pos,
  rivers: RiverSegment[]
): Pos | null {
  if (!rivers || rivers.length === 0) return null;

  const p1x = getX(st1), p1y = getY(st1);
  const p2x = getX(st2), p2y = getY(st2);

  for (const r of rivers) {
    const from = r.from ?? r.From;
    const to = r.to ?? r.To;
    if (!from || !to) continue;

    const rx1 = getX(from), ry1 = getY(from);
    const rx2 = getX(to), ry2 = getY(to);

    const hit = lineIntersection(p1x, p1y, p2x, p2y, rx1, ry1, rx2, ry2);
    if (hit) return hit;
  }

  return null;
}

function lineIntersection(
  x1: number, y1: number, x2: number, y2: number,
  x3: number, y3: number, x4: number, y4: number
): Pos | null {
  const denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
  if (denom === 0) return null;

  const ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
  const ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;

  if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
    return {
      x: x1 + ua * (x2 - x1),
      y: y1 + ua * (y2 - y1),
    };
  }
  return null;
}
