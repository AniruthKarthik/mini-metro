import type { StationDTO, StationKind } from '../types';
import { StationKind as StationKindValue, getX, getY } from '../types';
import { Viewport } from './viewport';
import { drawStationShape, drawPassengerShape, DARK_CHARCOAL, WHITE_FILL } from './shapes';

export function renderStations(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  stations: StationDTO[],
  hoveredStationId: number | null = null
): void {
  ctx.save();

  for (const st of stations) {
    if (!st.alive) continue;

    const screenPos = viewport.mapToScreen({ x: getX(st), y: getY(st) });
    const isHovered = st.id === hoveredStationId;
    const baseRadius = isHovered ? 15 : 13;

    // Magnetic Snap Ring Halo on Hover / Drag Target
    if (isHovered) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, baseRadius + 8, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(37, 37, 37, 0.22)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }

    if (st.overcrowding_timer > 0) {
      renderOvercrowdingTimer(ctx, screenPos.x, screenPos.y, baseRadius + 10, st.overcrowding_timer);
    }

    if (st.is_interchange) {
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, baseRadius + 7, 0, Math.PI * 2);
      ctx.fillStyle = WHITE_FILL;
      ctx.fill();
      ctx.strokeStyle = DARK_CHARCOAL;
      ctx.lineWidth = 3.5;
      ctx.stroke();
    }

    drawStationShape(
      ctx,
      st.kind,
      screenPos.x,
      screenPos.y,
      baseRadius,
      DARK_CHARCOAL,
      WHITE_FILL,
      3
    );

    if (st.queue_size > 0) {
      renderPassengerQueue(ctx, screenPos.x, screenPos.y, baseRadius, st.queue_destinations || [], st.queue_size);
    }
  }

  ctx.restore();
}

function renderOvercrowdingTimer(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  timerValue: number
): void {
  ctx.save();

  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
  ctx.lineWidth = 3;
  ctx.stroke();

  const maxTimer = 20.0;
  const progress = Math.max(0, Math.min(1, 1 - timerValue / maxTimer));
  const startAngle = -Math.PI / 2;
  const endAngle = startAngle + progress * Math.PI * 2;

  ctx.beginPath();
  ctx.arc(x, y, radius, startAngle, endAngle);
  ctx.strokeStyle = progress > 0.7 ? '#e64b3c' : DARK_CHARCOAL;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.stroke();

  ctx.restore();
}

function renderPassengerQueue(
  ctx: CanvasRenderingContext2D,
  stationX: number,
  stationY: number,
  stationRadius: number,
  destinations: StationKind[],
  queueSize: number
): void {
  ctx.save();

  const startX = stationX + stationRadius + 12;
  const startY = stationY - 4;
  const spacing = 10;
  const maxPerRow = 6;
  const total = Math.min(queueSize, 18);

  for (let i = 0; i < total; i++) {
    const col = i % maxPerRow;
    const row = Math.floor(i / maxPerRow);

    const px = startX + col * spacing;
    const py = startY + row * spacing;

    const destination = destinations[i] ?? StationKindValue.Circle;
    drawPassengerShape(ctx, destination, px, py, 3.7, DARK_CHARCOAL);
  }

  ctx.restore();
}
