import type { StationDTO, StationKind } from '../types';
import { getX, getY } from '../types';
import { Viewport } from './viewport';
import { drawStationShape, drawPassengerShape, DARK_CHARCOAL, WHITE_FILL } from './shapes';

export function renderStations(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  stations: StationDTO[],
  hoveredStationId: number | null
): void {
  ctx.save();

  for (const st of stations) {
    if (!st.alive) continue;

    const screenPos = viewport.mapToScreen({ x: getX(st), y: getY(st) });
    const isHovered = st.id === hoveredStationId;
    const baseRadius = isHovered ? 13 : 11;

    // Magnetic station snap target halo
    if (isHovered) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, baseRadius + 10, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.restore();
    }

    // Overcrowding progress ring
    if (st.overcrowding_timer > 0) {
      renderOvercrowdingTimer(ctx, screenPos.x, screenPos.y, baseRadius + 5, st.overcrowding_timer);
    }

    // Interchange Hub double outer capsule ring
    if (st.is_interchange) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, baseRadius + 6.5, 0, Math.PI * 2);
      ctx.strokeStyle = DARK_CHARCOAL;
      ctx.lineWidth = 3.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, baseRadius + 4.0, 0, Math.PI * 2);
      ctx.fillStyle = WHITE_FILL;
      ctx.fill();
      ctx.restore();
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
  stX: number,
  stY: number,
  stRadius: number,
  destinations: StationKind[],
  queueSize: number
): void {
  const shapeSize = 4.2;
  const spacing = 11;
  const startOffsetX = stRadius + 9;
  const startOffsetY = -stRadius + 2;

  const countToDraw = Math.min(destinations.length, 8);

  ctx.save();

  for (let i = 0; i < countToDraw; i++) {
    const kind = destinations[i];
    const px = stX + startOffsetX + (i % 4) * spacing;
    const py = stY + startOffsetY + Math.floor(i / 4) * spacing;

    drawPassengerShape(
      ctx,
      kind,
      px,
      py,
      shapeSize,
      DARK_CHARCOAL
    );
  }

  if (queueSize > 8) {
    ctx.fillStyle = DARK_CHARCOAL;
    ctx.font = '600 10px Avenir Next, Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`+${queueSize - 8}`, stX + startOffsetX + 4 * spacing, stY + startOffsetY + 5);
  }

  ctx.restore();
}
