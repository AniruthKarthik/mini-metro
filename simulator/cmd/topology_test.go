package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestAdjacencyGraph(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Star, Pos: engine.Pos{X: 30, Y: 0}},
	})

	sim.State.Resources.Grant(engine.RewardLine)

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{2, 3}})

	sim.Step(0.01)

	graph := sim.State.Graph
	if len(graph.Neighbours(1)) == 0 {
		t.Errorf("expected station 1 to have neighbours in graph")
	}

	lines01 := graph.LinesFor(0, 1)
	if len(lines01) != 1 || lines01[0] != 0 {
		t.Errorf("expected line 0 connecting stations 0 and 1, got %v", lines01)
	}

	lines23 := graph.LinesFor(2, 3)
	if len(lines23) != 1 || lines23[0] != 1 {
		t.Errorf("expected line 1 connecting stations 2 and 3, got %v", lines23)
	}
}

func TestTopologyVersionIncrement(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	initialVersion := sim.State.TopologyVersion
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})

	if sim.State.TopologyVersion <= initialVersion {
		t.Errorf("expected TopologyVersion to increment on AddLine, initial=%d new=%d", initialVersion, sim.State.TopologyVersion)
	}
}

func TestInsertStationIncrementsTopologyVersion(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 2}})
	before := sim.State.TopologyVersion

	if err := sim.ApplyAction(engine.InsertStation{LineID: 0, StationID: 1, Index: 1}); err != nil {
		t.Fatalf("InsertStation failed: %v", err)
	}
	if sim.State.TopologyVersion != before+1 {
		t.Fatalf("expected topology version %d, got %d", before+1, sim.State.TopologyVersion)
	}
}

func TestCloseLoopRequiresThreeStations(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err := sim.ApplyAction(engine.CloseLoop{LineID: 0}); err == nil {
		t.Fatalf("expected two-station loop closure to fail")
	}
	_ = sim.ApplyAction(engine.ExtendLine{LineID: 0, StationID: 2})
	if err := sim.ApplyAction(engine.CloseLoop{LineID: 0}); err != nil {
		t.Fatalf("expected three-station loop closure to succeed: %v", err)
	}
}
