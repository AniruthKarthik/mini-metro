package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestResourcePool(t *testing.T) {
	pool := engine.NewResourcePool()
	if pool.Lines != 3 {
		t.Errorf("expected 3 lines, got %d", pool.Lines)
	}
	if pool.Trains != 3 {
		t.Errorf("expected 3 trains, got %d", pool.Trains)
	}
	if pool.Tunnels != 0 {
		t.Errorf("expected 0 tunnels, got %d", pool.Tunnels)
	}
	if pool.Carriages != 0 {
		t.Errorf("expected 0 carriages, got %d", pool.Carriages)
	}

	if !pool.Spend(engine.RewardLine) {
		t.Errorf("failed to spend line")
	}
	if pool.Lines != 2 {
		t.Errorf("expected 2 lines remaining, got %d", pool.Lines)
	}

	pool.Grant(engine.RewardLine)
	if pool.Lines != 3 {
		t.Errorf("expected 3 lines after grant, got %d", pool.Lines)
	}
}

func TestActionGatingOnResources(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Star, Pos: engine.Pos{X: 30, Y: 0}},
		{ID: 4, Kind: engine.Pentagon, Pos: engine.Pos{X: 40, Y: 0}},
	})

	// Spend all 3 initial lines
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{1, 2}})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{2, 3}})

	// 4th line attempt should fail
	err := sim.ApplyAction(engine.AddLine{Stations: []int{3, 4}})
	if err == nil {
		t.Errorf("expected error when spending line from empty pool, got nil")
	}

	// Spend 3 trains
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 1})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 2})

	// 4th train attempt should fail
	err = sim.ApplyAction(engine.AddTrain{LineID: 0})
	if err == nil {
		t.Errorf("expected error when spending train from empty pool, got nil")
	}

	// Carriage attempt should fail (0 initial carriages)
	err = sim.ApplyAction(engine.AddCarriage{TrainID: 0})
	if err == nil {
		t.Errorf("expected error when spending carriage from empty pool, got nil")
	}
}

func TestResourceReturnOnRemoval(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	if sim.State.Resources.Lines != 2 {
		t.Errorf("expected 2 lines remaining before removal, got %d", sim.State.Resources.Lines)
	}

	err := sim.ApplyAction(engine.RemoveLine{LineID: 0})
	if err != nil {
		t.Fatalf("RemoveLine failed: %v", err)
	}

	if sim.State.Resources.Lines != 3 {
		t.Errorf("expected 3 lines after line removal refund, got %d", sim.State.Resources.Lines)
	}
	if sim.State.Resources.Trains != 3 {
		t.Errorf("expected 3 trains after line removal refund, got %d", sim.State.Resources.Trains)
	}
	if sim.State.Trains[0].Active {
		t.Errorf("expected train 0 to be deactivated on line removal")
	}
}
