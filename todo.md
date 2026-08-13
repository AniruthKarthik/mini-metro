# Mini Metro Frontend Implementation Plan (`/ui`)

This document outlines the step-by-step roadmap for implementing the Mini Metro frontend interface in the `ui/` directory at the repository root. The goal is an exact aesthetic and functional replication of the Mini Metro game interface, connected via WebSockets to the Go backend simulator (`simulator/cmd/server`).

---

## 📋 Task Checklist

### Phase 1: Project Setup & Architecture (`/ui`)
- [x] **1.1 Initialize Web Project**
  - Scaffold Vite + TypeScript application inside `ui/` folder (`npx -y create-vite@latest ui --template vanilla-ts`).
  - Configure build scripts, dev server on port `3000`, and WebSocket proxy/connection setup.
- [x] **1.2 Design System & Aesthetics**
  - Implement Mini Metro color palette:
    - Background: Off-white canvas (`#f4f3f0`)
    - Water fill: Light cyan (`#a5dcf5`) with smooth border strokes
    - Metro line colors: Standard Mini Metro palette (Red `#f53649`, Blue `#1b68db`, Green `#10b981`, Yellow `#f59e0b`, Pink `#ec4899`, Cyan `#06b6d4`, Purple `#8b5cf6`, Brown `#92400e`)
    - Station & Passenger glyphs: Charcoal stroke (`#22252a`) with crisp vector rendering
    - Font family: Clean modern sans-serif (`Inter`).
- [x] **1.3 WebSocket Client & Engine State Synchronization**
  - Implement robust WebSocket manager (`ws://localhost:6969/ws`).
  - Listen for `StateSnapshot` messages (tick, score, stations, lines, trains, rivers, water_polygons, resources, pending_reward_choices, alive, paused).
  - Implement state store with auto-reconnection and event dispatching for actions (`add_line`, `extend_line`, `add_train`, `remove_line`, `choose_reward`, `add_carriage`, `upgrade_interchange`, `shorten_line`, `close_loop`, `open_loop`, `reposition_train`, `pause`, `resume`, `set_speed`).
  - Toast/alert system for `action_error` feedback.

---

### Phase 2: Canvas Rendering Engine
- [x] **2.1 Multi-Layer Canvas / WebGL Architecture**
  - Set up high-performance Canvas 2D rendering pipeline:
    - Layer 0: Background
    - Layer 1: Water Geography (Rivers & Polygons)
    - Layer 2: Metro Line Paths (Octilinear 45° bends, smooth Bezier join curves, end-caps)
    - Layer 3: Station Nodes & Overcrowding Timers
    - Layer 4: Passenger Queues
    - Layer 5: Trains, Carriages & Passengers Onboard
    - Layer 6: Drag-and-Drop Interactive Overlay & Guides
- [x] **2.2 River & Geography Renderer**
  - Draw river paths (`RiverSegment`) with smooth curved strokes and proper cap joins.
  - Draw water polygons (`WaterPolygon`) for bays/lakes matching map definitions.
- [x] **2.3 Station Renderer**
  - Render 10 geometric station types: `Circle`, `Triangle`, `Square`, `Star`, `Pentagon`, `Gem`, `Sector`, `Cross`, `Drop`, `Oval`.
  - Draw crisp charcoal outlines with white interior fill.
  - Render double-ring / hub badge for Interchange upgraded stations (`is_interchange`).
- [x] **2.4 Passenger Queue & Overcrowding Renderer**
  - Draw passenger queue beside stations using mini geometric shape icons representing destination station types.
  - Render radial progress ring / circular overcrowding timer surrounding crowded stations (`overcrowding_timer > 0`).
- [x] **2.5 Metro Lines & Bezier Curves**
  - Render line paths with octilinear (45° angle) constraints or smooth Bezier curve routing between station coordinates.
  - Render line end-bars (T-shaped line end caps).
  - Indicate tunnels/bridges where line segments cross river geography (`tunnel_at`).
  - Render closed loop lines with seamless connection visualizer.
- [x] **2.6 Trains, Carriages & Smooth Interpolation**
  - Render pill-shaped train cars and trailing carriages in matching line color.
  - Implement 60 FPS smooth position interpolation (`Segment`, `Progress`, `Direction`) between 30 TPS WebSocket snapshot updates.
  - Display onboard passenger counts / passenger shape glyphs inside/above trains.

---

### Phase 3: Drag-and-Drop Line & Resource Interaction
- [x] **3.1 Interactive Line Drawing & Editing**
  - Click & drag from available line inventory tokens on the right panel to start a new line at any station.
  - Drag line end-cap to extend line to adjacent stations with line-snapping preview.
  - Drag line node off a station or drag end-cap to shorten/remove line.
  - Drag end-cap back to start station to trigger `close_loop`.
- [x] **3.2 Train & Carriage Allocation**
  - Drag train icon from left inventory dock and drop onto active metro line segment to dispatch train (`add_train`).
  - Drag carriage icon onto active train to attach extra carriage (`add_carriage`).
  - Drag train off line to reposition or reallocate (`reposition_train`).
- [x] **3.3 Station Interchange Upgrading**
  - Drag interchange token or click station to upgrade station to high-capacity interchange (`upgrade_interchange`).

---

### Phase 4: HUD & UI Overlays (Exact Screenshot Match)
- [x] **4.1 Top Header & Indicators**
  - Top-left: Back / Menu arrow icon (`←`).
  - Top-right:
    - Day of week text (e.g., `MON`, `TUE`, ..., `SAT`).
    - Cumulative passenger transport counter (e.g., `859`).
    - Animated analog clock icon ticking in sync with game simulation ticks.
- [x] **4.2 Right Control Panel (Speed & Line Inventory)**
  - Pause button (`||`) -> dispatches `pause` action.
  - 1x Play button (`>`) -> dispatches `resume` / `set_speed: {tps: 30}`.
  - 2x Fast-forward button (`>>`) -> dispatches `set_speed: {tps: 60}`.
  - Line Inventory Dock: Vertical stack of colored circular line tokens showing active lines and available lines count.
- [x] **4.3 Left Resource Dock**
  - Dark circular dock buttons:
    - Locomotive / Train icon with available count badge.
    - Carriage icon with available count badge.
    - Bridge / Tunnel icon with available count badge.
- [x] **4.4 Weekly Reward Modal Overlay**
  - Trigger modal when `pending_reward_choices` is non-empty.
  - Blur/darken background canvas.
  - Present 2 reward choices with Mini Metro styled iconography (+1 Line, +1 Train, +1 Tunnel, +1 Carriage, +1 Interchange).
  - Dispatch `choose_reward` WS action upon user click.
- [x] **4.5 Game Over Modal Screen**
  - Display game over banner when `alive == false`.
  - Show final passenger count, total days survived, map statistics.
  - "Restart Game" button to re-initialize simulator.

---

### Phase 5: Verification & Polish
- [x] **5.1 End-to-End WebSocket Testing**
  - Launch Go backend server (`go run simulator/cmd/server/main.go -map london`).
  - Launch Vite frontend (`npm run dev`) and test full gameplay loop.
- [x] **5.2 Responsive Viewport Scaling & HiDPI Support**
  - High-DPI (Retina) canvas pixel ratio scaling.
  - Responsive resizing to fit browser window with clean aspect ratio maintenance.
