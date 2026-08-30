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
	if sim.State.Resources.Lines != 2 {
		t.Errorf("expected 2 lines remaining before removal, got %d", sim.State.Resources.Lines)
	}

	_ = sim.ApplyAction(engine.RemoveLine{LineID: 0})
	if sim.State.Resources.Lines != 3 {
		t.Errorf("expected 3 lines after line removal refund, got %d", sim.State.Resources.Lines)
	}
}

func TestChooseRewardByPositionalIndexAndEnum(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
	})

	// BUG-1 fix: chooseReward now enforces strictly positional indices (0 or 1).
	// Set pending choices: [RewardTunnel (2), RewardCarriage (3)]
	sim.State.PendingRewardChoices = []engine.RewardType{engine.RewardTunnel, engine.RewardCarriage}

	// 1. Select by positional index 1 → should grant RewardCarriage.
	initialCarriages := sim.State.Resources.Carriages
	err := sim.ApplyAction(engine.ChooseReward{Choice: 1})
	if err != nil {
		t.Fatalf("unexpected error choosing reward by index 1: %v", err)
	}
	if sim.State.Resources.Carriages != initialCarriages+1 {
		t.Errorf("expected carriage count to increase by 1")
	}

	// 2. Set pending choices again: [RewardLine (0), RewardTunnel (2)]
	sim.State.PendingRewardChoices = []engine.RewardType{engine.RewardLine, engine.RewardTunnel}
	initialLines := sim.State.Resources.Lines

	// Select by positional index 0 → should grant RewardLine.
	err = sim.ApplyAction(engine.ChooseReward{Choice: engine.RewardType(0)})
	if err != nil {
		t.Fatalf("unexpected error choosing reward by index 0: %v", err)
	}
	if sim.State.Resources.Lines != initialLines+1 {
		t.Errorf("expected line count to increase by 1")
	}

	// 3. Out-of-range index should be rejected.
	sim.State.PendingRewardChoices = []engine.RewardType{engine.RewardLine, engine.RewardTunnel}
	err = sim.ApplyAction(engine.ChooseReward{Choice: engine.RewardType(2)})
	if err == nil {
		t.Error("expected error for out-of-range choice index 2, got nil")
	}
}

func TestTrainSlotReuseOnLineRemovalAndAdd(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	// Add line 0 (spawns train ID 0)
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if len(sim.State.Trains) != 1 {
		t.Fatalf("expected 1 train, got %d", len(sim.State.Trains))
	}

	// Remove line 0 (train 0 becomes inactive)
	_ = sim.ApplyAction(engine.RemoveLine{LineID: 0})
	if sim.State.Trains[0].Active {
		t.Fatalf("expected train 0 to be inactive after line removal")
	}

	// Add line 0 again (should reuse train ID 0 slot instead of appending a new train)
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if len(sim.State.Trains) != 1 {
		t.Errorf("expected train slot reuse (train count 1), got %d trains", len(sim.State.Trains))
	}
	if !sim.State.Trains[0].Active {
		t.Errorf("expected train 0 to be reactivated")
	}
}

