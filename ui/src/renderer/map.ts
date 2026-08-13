import type { RiverSegment, WaterPolygon } from '../types';
import { Viewport } from './viewport';

export const WATER_FILL = '#a5dcf5';
export const WATER_STROKE = '#7cc8ee';
export const BACKGROUND_COLOR = '#f4f3f0';

export function renderMapWater(
  ctx: CanvasRenderingContext2D,
  viewport: Viewport,
  rivers: RiverSegment[],
  waterPolygons: WaterPolygon[]
): void {
  ctx.save();

  if (waterPolygons && waterPolygons.length > 0) {
    for (const poly of waterPolygons) {
      const vertices = poly.vertices || poly.Vertices;
      if (!vertices || vertices.length < 3) continue;

      ctx.beginPath();
      const p0 = viewport.mapToScreen(vertices[0]);
      ctx.moveTo(p0.x!, p0.y!);

      for (let i = 1; i < vertices.length; i++) {
        const p = viewport.mapToScreen(vertices[i]);
        ctx.lineTo(p.x!, p.y!);
      }
      ctx.closePath();

      ctx.fillStyle = WATER_FILL;
      ctx.fill();
      ctx.strokeStyle = WATER_STROKE;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  if (rivers && rivers.length > 0) {
    for (const river of rivers) {
      const fromPos = river.from || river.From;
      const toPos = river.to || river.To;
      if (!fromPos || !toPos) continue;

      const pFrom = viewport.mapToScreen(fromPos);
      const pTo = viewport.mapToScreen(toPos);
      const widthVal = river.width ?? river.Width ?? 6;
      const pixelWidth = viewport.mapDistToScreen(widthVal);

      ctx.beginPath();
      ctx.moveTo(pFrom.x!, pFrom.y!);
      ctx.lineTo(pTo.x!, pTo.y!);
      ctx.strokeStyle = WATER_STROKE;
      ctx.lineWidth = pixelWidth + 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(pFrom.x!, pFrom.y!);
      ctx.lineTo(pTo.x!, pTo.y!);
      ctx.strokeStyle = WATER_FILL;
      ctx.lineWidth = pixelWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
  }

  ctx.restore();
}
