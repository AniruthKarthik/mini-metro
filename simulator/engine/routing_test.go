package engine

import (
	"testing"
)

func TestFindOptimalRoute_PrioritizesDirectOverTransfer(t *testing.T) {
	// Setup 3 stations:
	// Station 0: Circle at (0, 0)
	// Station 1: Triangle at (10, 0)
	// Station 2: Square at (20, 0)
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	}
	sim := NewSimulator(stations)

	// Line 0 connects 0 -> 1 -> 2 (Direct connection from 0 to 2)
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})

	// Line 1 connects 0 -> 1
	// Line 2 connects 1 -> 2
	// A route from 0 to 2 using Line 1 then Line 2 would require 1 transfer.
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(AddLine{Stations: []int{1, 2}})

	sim.rebuildGraphIfNeeded()

	// Find route from Station 0 to Square (Station 2)
	route := FindOptimalRoute(&sim.State.Graph, &sim.State, 0, Square)

	if !route.Reachable {
		t.Fatalf("expected route to be reachable")
	}
	if !route.IsDirect {
		t.Errorf("expected direct route, got transfers=%d", route.Transfers)
	}
	if route.Transfers != 0 {
		t.Errorf("expected 0 transfers, got %d", route.Transfers)
	}
	if route.NextLineID != 0 {
		t.Errorf("expected NextLineID == 0 (direct line), got %d", route.NextLineID)
	}
}

func TestFindOptimalRoute_PrefersFewerTransfers(t *testing.T) {
	// Setup 4 stations: 0 (Circle), 1 (Triangle), 2 (Cross), 3 (Square)
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Cross, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
		{ID: 3, Kind: Square, Pos: Pos{X: 30, Y: 0}, Alive: true, Capacity: 6},
	}
	sim := NewSimulator(stations)

	// Route A (1 transfer):
	// Line 0: 0 -> 1 -> 2
	// Line 1: 2 -> 3
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})
	_ = sim.ApplyAction(AddLine{Stations: []int{2, 3}})

	sim.rebuildGraphIfNeeded()

	route := FindOptimalRoute(&sim.State.Graph, &sim.State, 0, Square)
	if !route.Reachable {
		t.Fatalf("expected route to be reachable")
	}
	if route.Transfers != 1 {
		t.Errorf("expected 1 transfer, got %d", route.Transfers)
	}
	if route.NextLineID != 0 {
		t.Errorf("expected NextLineID == 0, got %d", route.NextLineID)
	}
}
