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

	sim.State.Resources.Grant(engine.RewardLine)

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
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})

	// Line only connects 0 and 1; 2 is unreachable
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	sim.Step(0.01)

	rUnreachable := engine.FindOptimalRoute(&sim.State.Graph, &sim.State, 0, engine.Square)
	if rUnreachable.Reachable {
		t.Errorf("expected station 2 (Square) to be unreachable from station 0")
	}
}
