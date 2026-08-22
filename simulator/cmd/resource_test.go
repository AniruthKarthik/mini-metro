package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestResourcePool(t *testing.T) {
	pool := engine.NewResourcePool()
	if pool.Lines != 1 {
		t.Errorf("expected 1 line, got %d", pool.Lines)
	}
	if pool.Trains != 1 {
		t.Errorf("expected 1 train, got %d", pool.Trains)
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
	if pool.Lines != 0 {
		t.Errorf("expected 0 lines remaining, got %d", pool.Lines)
	}

	pool.Grant(engine.RewardLine)
	if pool.Lines != 1 {
		t.Errorf("expected 1 line after grant, got %d", pool.Lines)
	}
}

func TestActionGatingOnResources(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	// Spend starting train to test gating
	sim.State.Resources.Spend(engine.RewardTrain)

	err := sim.ApplyAction(engine.AddTrain{LineID: 0})
	if err == nil {
		t.Errorf("expected error when spending train from empty pool, got nil")
	}

	sim.State.Resources.Grant(engine.RewardTrain)
	sim.State.Resources.Grant(engine.RewardTrain)
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	err = sim.ApplyAction(engine.AddTrain{LineID: 0})
	if err != nil {
		t.Errorf("unexpected error adding train after grant: %v", err)
	}
}

func TestResourceReturnOnRemoval(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if sim.State.Resources.Lines != 0 {
		t.Errorf("expected 0 lines remaining before removal, got %d", sim.State.Resources.Lines)
	}

	_ = sim.ApplyAction(engine.RemoveLine{LineID: 0})
	if sim.State.Resources.Lines != 1 {
		t.Errorf("expected 1 line after line removal refund, got %d", sim.State.Resources.Lines)
	}
}
