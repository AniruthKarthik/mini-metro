package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestAPathfindingDirectAndTransferRoutes(t *testing.T) {
	// 0 (Circle) --Line 0-- 1 (Triangle) --Line 1-- 2 (Square)
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{1, 2}})

	sim.Step(0.01)

	// Direct route check: 0 -> 1 (Triangle)
	rDirect := engine.FindOptimalRoute(&sim.State.Graph, &sim.State, 0, engine.Triangle)
	if !rDirect.Reachable {
		t.Errorf("expected station 0 to reach Triangle")
	}
	if !rDirect.IsDirect {
		t.Errorf("expected route 0 -> Triangle to be direct")
	}
	if rDirect.NextLineID != 0 {
		t.Errorf("expected NextLineID = 0, got %d", rDirect.NextLineID)
	}

	// Transfer route check: 0 -> 2 (Square) via 1
	rTransfer := engine.FindOptimalRoute(&sim.State.Graph, &sim.State, 0, engine.Square)
	if !rTransfer.Reachable {
		t.Errorf("expected station 0 to reach Square via transfer")
	}
	if rTransfer.IsDirect {
		t.Errorf("expected transfer route 0 -> Square to NOT be direct")
	}
	if rTransfer.NextLineID != 0 {
		t.Errorf("expected initial line = 0, got %d", rTransfer.NextLineID)
	}
}

func TestUnreachablePassengerFiltering(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Pentagon, Pos: engine.Pos{X: 50, Y: 50}}, // Isolated Pentagon
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	// Seed queue with unreachable passenger targeting Pentagon
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, engine.Passenger{Destination: engine.Pentagon})

	sim.Step(0.01)

	// Verify unreachable passenger remained in station queue and did not board
	queueLen := len(sim.State.Stations[0].Queue)
	if queueLen == 0 {
		t.Errorf("expected unreachable passenger to remain in station queue")
	}
}
