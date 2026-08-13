import type { LineDTO, StationDTO, Pos } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { LINE_COLORS } from './shapes';

export function getLineColor(lineId: number): string {
  return LINE_COLORS[lineId % LINE_COLORS.length];
}

// Converts raw station coordinates into a 45-degree octilinear path
export function generateOctilinearPath(points: Pos[]): Pos[] {
  if (points.length < 2) return points;

  const result: Pos[] = [points[0]];

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = result[result.length - 1];
    const p2 = points[i + 1];

    const x1 = getX(p1), y1 = getY(p1);
    const x2 = getX(p2), y2 = getY(p2);

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);

    // If already horizontal, vertical, or 45-degree diagonal
    if (dx < 4 || dy < 4 || Math.abs(dx - dy) < 4) {
      result.push(p2);
      continue;
    }

    // Calculate intermediate 45-degree chamfer corner point
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
  activeDragLinePreview?: { points: Pos[]; color: string }
): void {
  ctx.save();

  const stationMap = new Map<number, StationDTO>();
  for (const st of stations) {
    stationMap.set(st.id, st);
  }

  // Draw active metro lines
  for (const line of lines) {
    if (line.removed || !line.stations || line.stations.length < 2) {
      continue;
    }

    const color = getLineColor(line.id);
    const rawScreenPoints: Pos[] = [];

    for (const stId of line.stations) {
      const st = stationMap.get(stId);
      if (st) {
        rawScreenPoints.push(viewport.mapToScreen({ x: getX(st), y: getY(st) }));
      }
    }

    if (line.is_loop && rawScreenPoints.length > 2) {
      rawScreenPoints.push(rawScreenPoints[0]);
    }

    if (rawScreenPoints.length < 2) continue;

    // Convert straight segments to octilinear 45° paths
    const octilinearPoints = generateOctilinearPath(rawScreenPoints);

    // Draw main track line with smooth rounded corners
    drawOctilinearTrack(ctx, octilinearPoints, color, 9.0);

    // Draw Tunnel / Bridge markers where tunnel_at is true
    if (line.tunnel_at) {
      for (let i = 0; i < line.tunnel_at.length; i++) {
        if (line.tunnel_at[i] && i < rawScreenPoints.length - 1) {
          drawBridgeMarker(ctx, rawScreenPoints[i], rawScreenPoints[i + 1]);
        }
      }
    }

    // Draw T-bar end caps for non-loop line terminals
    if (!line.is_loop && rawScreenPoints.length >= 2) {
      drawTerminalEndCap(ctx, rawScreenPoints[0], rawScreenPoints[1], color, 9.0);
      const lastIdx = rawScreenPoints.length - 1;
      drawTerminalEndCap(ctx, rawScreenPoints[lastIdx], rawScreenPoints[lastIdx - 1], color, 9.0);
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
  lineWidth: number = 9.0,
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
    ctx.setLineDash([8, 6]);
  }

  ctx.stroke();
  ctx.restore();
}

function drawTerminalEndCap(
  ctx: CanvasRenderingContext2D,
  terminal: Pos,
  nextPt: Pos,
  color: string,
  lineWidth: number
): void {
  const termX = getX(terminal), termY = getY(terminal);
  const nextX = getX(nextPt), nextY = getY(nextPt);

  const dx = nextX - termX;
  const dy = nextY - termY;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return;

  const nx = -dy / len;
  const ny = dx / len;

  const capWidth = lineWidth * 1.6;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(termX - nx * capWidth, termY - ny * capWidth);
  ctx.lineTo(termX + nx * capWidth, termY + ny * capWidth);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth * 0.95;
  ctx.lineCap = 'butt';
  ctx.stroke();
  ctx.restore();
}

function drawBridgeMarker(ctx: CanvasRenderingContext2D, p1: Pos, p2: Pos): void {
  const p1x = getX(p1), p1y = getY(p1);
  const p2x = getX(p2), p2y = getY(p2);

  const midX = (p1x + p2x) / 2;
  const midY = (p1y + p2y) / 2;

  const dx = p2x - p1x;
  const dy = p2y - p1y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return;

  const nx = -dy / len;
  const ny = dx / len;

  const bridgeW = 11;
  const gap = 4;

  ctx.save();
  ctx.strokeStyle = '#22252a';
  ctx.lineWidth = 2.8;

  ctx.beginPath();
  ctx.moveTo(midX - (dx / len) * gap - nx * bridgeW, midY - (dy / len) * gap - ny * bridgeW);
  ctx.lineTo(midX - (dx / len) * gap + nx * bridgeW, midY - (dy / len) * gap + ny * bridgeW);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(midX + (dx / len) * gap - nx * bridgeW, midY + (dy / len) * gap - ny * bridgeW);
  ctx.lineTo(midX + (dx / len) * gap + nx * bridgeW, midY + (dy / len) * gap + ny * bridgeW);
  ctx.stroke();

  ctx.restore();
}
