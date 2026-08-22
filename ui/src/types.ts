export const StationKind = {
  Circle: 0,
  Triangle: 1,
  Square: 2,
  Star: 3,
  Pentagon: 4,
  Gem: 5,
  Sector: 6,
  Cross: 7,
  Drop: 8,
  Oval: 9,
} as const;

export type StationKind = typeof StationKind[keyof typeof StationKind];

export interface Pos {
  x?: number;
  y?: number;
  X?: number;
  Y?: number;
}

export function getX(p?: Pos): number {
  if (!p) return 0;
  return p.x ?? p.X ?? 0;
}

export function getY(p?: Pos): number {
  if (!p) return 0;
  return p.y ?? p.Y ?? 0;
}

export interface RiverSegment {
  from?: Pos;
  to?: Pos;
  From?: Pos;
  To?: Pos;
  width?: number;
  Width?: number;
}

export interface WaterPolygon {
  vertices?: Pos[];
  Vertices?: Pos[];
}

export interface ResourcePool {
  lines: number;
  trains: number;
  tunnels: number;
  carriages: number;
  interchanges: number;
}

export const RewardType = {
  RewardLine: 0,
  RewardTrain: 1,
  RewardTunnel: 2,
  RewardCarriage: 3,
  RewardInterchange: 4,
} as const;

export type RewardType = typeof RewardType[keyof typeof RewardType];

export interface StationDTO {
  id: number;
  kind: StationKind;
  kind_name: string;
  x: number;
  y: number;
  queue_size: number;
  queue_destinations?: StationKind[];
  capacity: number;
  overcrowding_timer: number; // -1 = no timer
  is_interchange: boolean;
  alive: boolean;
}

export interface LineDTO {
  id: number;
  stations: number[];
  tunnel_at: boolean[];
  is_loop: boolean;
  loop_tunnel: boolean;
  removed: boolean;
}

export interface TrainDTO {
  id: number;
  line_id: number;
  segment: number;
  progress: number;
  direction: number;
  capacity: number;
  carriages: number;
  load: number;
  passengers?: StationKind[];
}

export interface StateSnapshot {
  tick: number;
  score: number;
  alive: boolean;
  map_name: string;
  paused: boolean;
  tps: number;
  stations: StationDTO[];
  lines: LineDTO[];
  trains: TrainDTO[];
  rivers: RiverSegment[];
  water_polygons: WaterPolygon[];
  resources: ResourcePool;
  pending_reward_choices: RewardType[];
  adjacency_list?: Record<number, number[]>;
}

export interface ActionEnvelope {
  type: string;
  payload?: any;
}

export type DragSource =
  | { type: 'new_line'; lineIndex: number }
  | { type: 'extend_line'; lineId: number; fromStationId: number; fromFront: boolean }
  | { type: 'add_train' }
  | { type: 'add_carriage' }
  | { type: 'reposition_train'; trainId: number };

export interface DragState {
  source: DragSource;
  currentPos: Pos;
  targetStationId: number | null;
}
