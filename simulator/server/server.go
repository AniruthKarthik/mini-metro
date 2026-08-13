package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

type StateSnapshot struct {
	// Meta
	Tick    uint64 `json:"tick"`
	Score   int    `json:"score"`
	Alive   bool   `json:"alive"`
	MapName string `json:"map_name"`
	Paused  bool   `json:"paused"`
	TPS     int    `json:"tps"` // current ticks-per-second
	// Stations (full detail)
	Stations []StationDTO `json:"stations"`
	// Lines (full detail)
	Lines []LineDTO `json:"lines"`
	// Trains (active only)
	Trains []TrainDTO `json:"trains"`
	// Water geography
	Rivers        []engine.RiverSegment `json:"rivers"`
	WaterPolygons []engine.WaterPolygon `json:"water_polygons"`
	// Resources + reward state
	Resources            engine.ResourcePool `json:"resources"`
	PendingRewardChoices []engine.RewardType `json:"pending_reward_choices"`
	// Adjacency for routing visualisation
	AdjacencyList map[int][]int `json:"adjacency_list,omitempty"`
}

// StationDTO carries the full station state for the renderer.
type StationDTO struct {
	ID                int                  `json:"id"`
	Kind              engine.StationKind   `json:"kind"`
	KindName          string               `json:"kind_name"`
	X                 float64              `json:"x"`
	Y                 float64              `json:"y"`
	QueueSize         int                  `json:"queue_size"`
	QueueDestinations []engine.StationKind `json:"queue_destinations"`
	Capacity          int                  `json:"capacity"`
	OvercrowdingTimer float64              `json:"overcrowding_timer"` // -1 = no timer
	IsInterchange     bool                 `json:"is_interchange"`
	Alive             bool                 `json:"alive"`
}

// LineDTO carries full line topology for the renderer to draw Bezier curves.
type LineDTO struct {
	ID         int    `json:"id"`
	Stations   []int  `json:"stations"`
	TunnelAt   []bool `json:"tunnel_at"`
	IsLoop     bool   `json:"is_loop"`
	LoopTunnel bool   `json:"loop_tunnel"`
	Removed    bool   `json:"removed"`
}

// TrainDTO carries per-train state for smooth interpolation on the client.
type TrainDTO struct {
	ID         int                  `json:"id"`
	LineID     int                  `json:"line_id"`
	Segment    int                  `json:"segment"`
	Progress   float64              `json:"progress"`
	Direction  int                  `json:"direction"`
	Capacity   int                  `json:"capacity"`
	Carriages  int                  `json:"carriages"`
	Load       int                  `json:"load"` // number of passengers on board
	Passengers []engine.StationKind `json:"passengers"`
}

// ErrorMessage is sent back to a specific client when an action fails.
type ErrorMessage struct {
	Type  string `json:"type"`
	Error string `json:"error"`
}

// Server wires together the hub, the game loop, and HTTP routes.
type Server struct {
	sim      *engine.Simulator
	hub      *Hub
	actionCh chan []byte
	mu       sync.Mutex
	paused   bool
	tps      int // ticks per second
}

// New constructs a Server around an already-initialised Simulator.
func New(sim *engine.Simulator) *Server {
	return &Server{
		sim:      sim,
		hub:      NewHub(),
		actionCh: make(chan []byte, 512),
		tps:      30, // default: 30 ticks / second
	}
}

// RegisterRoutes attaches the WebSocket and health-check endpoints to mux.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		ServeWs(s.hub, s.actionCh, w, r)
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})
}

// Run starts the hub, the action dispatcher, and the simulation game-loop.
// It blocks until ctx is done (or forever if you pass a background context).
func (s *Server) Run() {
	go s.hub.Run()
	go s.actionDispatcher()
	s.gameLoop()
}

// gameLoop advances the simulation at `s.tps` ticks per second and broadcasts
// a StateSnapshot after every tick.
func (s *Server) gameLoop() {
	for {
		s.mu.Lock()
		tps := s.tps
		paused := s.paused
		s.mu.Unlock()
		if paused || !s.sim.State.Alive {
			// Still broadcast while paused / game-over so the UI can react.
			s.broadcastState(paused)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		dt := 1.0 / 30.0 // Standard 30 TPS simulation step quantum
		s.sim.Step(dt)
		s.broadcastState(false)
		time.Sleep(time.Duration(float64(time.Second) / float64(tps)))
	}
}

// broadcastState serialises the current simulation state and sends it to all clients.
func (s *Server) broadcastState(paused bool) {
	snap := s.buildSnapshot(paused)
	data, err := json.Marshal(snap)
	if err != nil {
		log.Printf("state marshal error: %v", err)
		return
	}
	s.hub.Broadcast(data)
}

// buildSnapshot converts the engine's internal GameState into a StateSnapshot.
func (s *Server) buildSnapshot(paused bool) StateSnapshot {
	st := &s.sim.State
	snap := StateSnapshot{
		Tick:                 st.Tick,
		Score:                st.Score,
		Alive:                st.Alive,
		MapName:              st.MapName,
		Paused:               paused,
		TPS:                  s.tps,
		Rivers:               st.Rivers,
		WaterPolygons:        st.WaterPolygons,
		Resources:            st.Resources,
		PendingRewardChoices: append([]engine.RewardType(nil), st.PendingRewardChoices...),
	}
	// Stations
	for _, station := range st.Stations {
		queueDestinations := make([]engine.StationKind, 0, len(station.Queue))
		for _, passenger := range station.Queue {
			queueDestinations = append(queueDestinations, passenger.Destination)
		}
		snap.Stations = append(snap.Stations, StationDTO{
			ID:                station.ID,
			Kind:              station.Kind,
			KindName:          station.Kind.String(),
			X:                 station.Pos.X,
			Y:                 station.Pos.Y,
			QueueSize:         len(station.Queue),
			QueueDestinations: queueDestinations,
			Capacity:          station.Capacity,
			OvercrowdingTimer: station.OvercrowdingTimer,
			IsInterchange:     station.IsInterchange,
			Alive:             station.Alive,
		})
	}
	// Lines
	for _, line := range st.Lines {
		tunnelAt := make([]bool, len(line.TunnelAt))
		copy(tunnelAt, line.TunnelAt)
		snap.Lines = append(snap.Lines, LineDTO{
			ID:         line.ID,
			Stations:   append([]int(nil), line.Stations...),
			TunnelAt:   tunnelAt,
			IsLoop:     line.IsLoop,
			LoopTunnel: line.LoopTunnel,
			Removed:    line.Removed,
		})
	}
	// Trains (active only)
	for _, tr := range st.Trains {
		if !tr.Active {
			continue
		}
		passengers := make([]engine.StationKind, 0, len(tr.Passengers))
		for _, passenger := range tr.Passengers {
			passengers = append(passengers, passenger.Destination)
		}
		snap.Trains = append(snap.Trains, TrainDTO{
			ID:         tr.ID,
			LineID:     tr.LineID,
			Segment:    tr.Segment,
			Progress:   tr.Progress,
			Direction:  tr.Direction,
			Capacity:   tr.Capacity,
			Carriages:  tr.Carriages,
			Load:       len(tr.Passengers),
			Passengers: passengers,
		})
	}
	// Adjacency list (copy so caller doesn't mutate cached graph)
	if len(st.Graph.Adj) > 0 {
		snap.AdjacencyList = make(map[int][]int, len(st.Graph.Adj))
		for k, v := range st.Graph.Adj {
			nb := make([]int, len(v))
			copy(nb, v)
			snap.AdjacencyList[k] = nb
		}
	}
	return snap
}

// actionDispatcher reads raw JSON from actionCh, parses it, and either
// applies engine actions or handles server-side controls.
func (s *Server) actionDispatcher() {
	for raw := range s.actionCh {
		action, cmd, err := ParseAction(raw)
		if err != nil {
			log.Printf("❌ Action parse error: %v (raw: %s)", err, raw)
			errMsg, _ := json.Marshal(ErrorMessage{
				Type:  "action_error",
				Error: err.Error(),
			})
			s.hub.Broadcast(errMsg)
			continue
		}
		if cmd != "" {
			s.handleServerCommand(cmd)
			continue
		}
		if err := s.sim.ApplyAction(action); err != nil {
			log.Printf("❌ Action apply error: %v (raw: %s)", err, raw)
			errMsg, _ := json.Marshal(ErrorMessage{
				Type:  "action_error",
				Error: err.Error(),
			})
			s.hub.Broadcast(errMsg)
		} else {
			log.Printf("✅ Action applied successfully: %s", string(raw))
		}
	}
}

// handleServerCommand processes non-engine commands like pause/resume/set_speed.
func (s *Server) handleServerCommand(cmd string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case cmd == "pause":
		s.paused = true
		log.Println("simulation paused")
	case cmd == "resume":
		s.paused = false
		log.Println("simulation resumed")
	case cmd == "restart":
		s.sim = engine.NewSimulatorWithMap(engine.LondonMap())
		s.paused = false
		log.Println("🔄 Simulation restarted cleanly with London map")
	case strings.HasPrefix(cmd, "set_speed:"):
		// payload is a raw JSON object {"tps":60}
		payload := strings.TrimPrefix(cmd, "set_speed:")
		var p struct {
			TPS int `json:"tps"`
		}
		if err := json.Unmarshal([]byte(payload), &p); err == nil && p.TPS > 0 && p.TPS <= 300 {
			s.tps = p.TPS
			log.Printf("simulation speed set to %d tps", s.tps)
		}
	default:
		log.Printf("unknown server command: %s", cmd)
	}
}
