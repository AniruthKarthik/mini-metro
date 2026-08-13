import { StationKind } from '../types';

export const LINE_COLORS = [
  '#f53649', // Red
  '#1b68db', // Blue
  '#10b981', // Green
  '#f59e0b', // Yellow
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#8b5cf6', // Purple
  '#92400e', // Brown
];

export const DARK_CHARCOAL = '#22252a';
export const WHITE_FILL = '#ffffff';

export function drawStationShape(
  ctx: CanvasRenderingContext2D,
  kind: StationKind,
  x: number,
  y: number,
  size: number,
  strokeColor: string = DARK_CHARCOAL,
  fillColor: string = WHITE_FILL,
  lineWidth: number = 3.5
): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();

  const r = size;

  switch (kind) {
    case StationKind.Circle:
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      break;

    case StationKind.Triangle: {
      const top = -r * 1.15;
      const bottom = r * 0.85;
      const width = r * 1.15;
      ctx.moveTo(0, top);
      ctx.lineTo(width, bottom);
      ctx.lineTo(-width, bottom);
      ctx.closePath();
      break;
    }

    case StationKind.Square: {
      const side = r * 1.6;
      ctx.rect(-side / 2, -side / 2, side, side);
      break;
    }

    case StationKind.Star: {
      const points = 5;
      const outerR = r * 1.25;
      const innerR = r * 0.55;
      for (let i = 0; i < points * 2; i++) {
        const radius = i % 2 === 0 ? outerR : innerR;
        const angle = (i * Math.PI) / points - Math.PI / 2;
        const px = Math.cos(angle) * radius;
        const py = Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }

    case StationKind.Pentagon: {
      const points = 5;
      const radius = r * 1.1;
      for (let i = 0; i < points; i++) {
        const angle = (i * 2 * Math.PI) / points - Math.PI / 2;
        const px = Math.cos(angle) * radius;
        const py = Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }

    case StationKind.Gem: {
      // Diamond shape
      ctx.moveTo(0, -r * 1.3);
      ctx.lineTo(r * 1.1, 0);
      ctx.lineTo(0, r * 1.3);
      ctx.lineTo(-r * 1.1, 0);
      ctx.closePath();
      break;
    }

    case StationKind.Sector: {
      // Pie sector (quarter circle)
      ctx.arc(0, 0, r * 1.1, -Math.PI / 2, 0);
      ctx.lineTo(0, 0);
      ctx.closePath();
      break;
    }

    case StationKind.Cross: {
      const arm = r * 1.1;
      const w = r * 0.38;
      ctx.moveTo(-w, -arm);
      ctx.lineTo(w, -arm);
      ctx.lineTo(w, -w);
      ctx.lineTo(arm, -w);
      ctx.lineTo(arm, w);
      ctx.lineTo(w, w);
      ctx.lineTo(w, arm);
      ctx.lineTo(-w, arm);
      ctx.lineTo(-w, w);
      ctx.lineTo(-arm, w);
      ctx.lineTo(-arm, -w);
      ctx.lineTo(-w, -w);
      ctx.closePath();
      break;
    }

    case StationKind.Drop: {
      // Water drop / Teardrop shape
      const dropR = r * 0.8;
      ctx.arc(0, dropR * 0.3, dropR, 0, Math.PI);
      ctx.quadraticCurveTo(-dropR, -dropR * 0.5, 0, -r * 1.2);
      ctx.quadraticCurveTo(dropR, -dropR * 0.5, dropR, dropR * 0.3);
      ctx.closePath();
      break;
    }

    case StationKind.Oval:
    default: {
      // Horizontal Ellipse
      ctx.ellipse(0, 0, r * 1.3, r * 0.8, 0, 0, Math.PI * 2);
      break;
    }
  }

  if (fillColor) {
    ctx.fillStyle = fillColor;
    ctx.fill();
  }

  if (strokeColor && lineWidth > 0) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  ctx.restore();
}

export function drawPassengerShape(
  ctx: CanvasRenderingContext2D,
  kind: StationKind,
  x: number,
  y: number,
  size: number = 4.5,
  color: string = DARK_CHARCOAL
): void {
  // Passengers are rendered as small solid filled glyphs
  drawStationShape(ctx, kind, x, y, size, color, color, 0);
}
