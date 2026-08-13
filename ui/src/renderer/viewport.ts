import type { Pos } from '../types';
import { getX, getY } from '../types';

export interface ScreenPos {
  x: number;
  y: number;
}

export class Viewport {
  public width: number = 800;
  public height: number = 600;
  public padding: number = 60;

  public updateSize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  public mapToScreen(pos: Pos): ScreenPos {
    const usableW = this.width - this.padding * 2;
    const usableH = this.height - this.padding * 2;

    const px = getX(pos);
    const py = getY(pos);

    const screenX = this.padding + (px / 100) * usableW;
    const screenY = this.padding + (py / 100) * usableH;

    return { x: screenX, y: screenY };
  }

  public mapDistToScreen(dist: number): number {
    const usableW = this.width - this.padding * 2;
    const usableH = this.height - this.padding * 2;
    const scale = (usableW + usableH) / 200;
    return dist * scale;
  }

  public screenToMap(pos: Pos): ScreenPos {
    const usableW = this.width - this.padding * 2;
    const usableH = this.height - this.padding * 2;

    const px = getX(pos);
    const py = getY(pos);

    const mapX = ((px - this.padding) / usableW) * 100;
    const mapY = ((py - this.padding) / usableH) * 100;

    return { x: mapX, y: mapY };
  }
}
