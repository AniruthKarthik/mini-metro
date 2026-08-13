import type { LineDTO, StationDTO, Pos } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { LINE_COLORS } from './shapes';

export function getLineColor(lineId: number): string {
  return LINE_COLORS[lineId % LINE_COLORS.length];
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

  for (const line of lines) {
    if (line.removed || !line.stations || line.stations.length < 2) {
      continue;
    }

    const color = getLineColor(line.id);
    const screenPoints: Pos[] = [];

    for (const stId of line.stations) {
      const st = stationMap.get(stId);
      if (st) {
        screenPoints.push(viewport.mapToScreen({ x: st.x, y: st.y }));
      }
    }

    if (line.is_loop && screenPoints.length > 2) {
      screenPoints.push(screenPoints[0]);
    }

    if (screenPoints.length < 2) continue;

    drawOctilinearTrack(ctx, screenPoints, color, 8.5);

    if (line.tunnel_at) {
      for (let i = 0; i < line.tunnel_at.length; i++) {
        if (line.tunnel_at[i] && i < screenPoints.length - 1) {
          drawBridgeMarker(ctx, screenPoints[i], screenPoints[i + 1]);
        }
      }
    }

    if (!line.is_loop) {
      drawTerminalEndCap(ctx, screenPoints[0], screenPoints[1], color, 8.5);
      const lastIdx = screenPoints.length - 1;
      drawTerminalEndCap(ctx, screenPoints[lastIdx], screenPoints[lastIdx - 1], color, 8.5);
    }
  }

  if (activeDragLinePreview && activeDragLinePreview.points.length >= 2) {
    drawOctilinearTrack(ctx, activeDragLinePreview.points, activeDragLinePreview.color, 7.0, true);
  }

  ctx.restore();
}

export function drawOctilinearTrack(
  ctx: CanvasRenderingContext2D,
  points: Pos[],
  color: string,
  lineWidth: number = 8.5,
  isDashed: boolean = false
): void {
  if (points.length < 2) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(getX(points[0]), getY(points[0]));

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];

    const p1x = getX(p1), p1y = getY(p1);
    const p2x = getX(p2), p2y = getY(p2);

    if (i < points.length - 2) {
      const p3 = points[i + 2];
      const p3x = getX(p3), p3y = getY(p3);
      const mid1X = (p1x + p2x) / 2;
      const mid1Y = (p1y + p2y) / 2;
      const mid2X = (p2x + p3x) / 2;
      const mid2Y = (p2y + p3y) / 2;

      ctx.lineTo(mid1X, mid1Y);
      ctx.quadraticCurveTo(p2x, p2y, mid2X, mid2Y);
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

  const capWidth = lineWidth * 1.5;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(termX - nx * capWidth, termY - ny * capWidth);
  ctx.lineTo(termX + nx * capWidth, termY + ny * capWidth);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth * 0.9;
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

  const bridgeW = 10;
  const gap = 4;

  ctx.save();
  ctx.strokeStyle = '#22252a';
  ctx.lineWidth = 2.5;

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
