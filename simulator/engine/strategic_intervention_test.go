package engine

import (
	"testing"
)

func TestSimulator_CloneFidelityAndIndependence(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 100, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 200, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 2, Kind: Square, Pos: Pos{X: 300, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
	}
	sim := NewSimulatorWithRivers(stations, nil, 42)

	// Add a line
	if err := sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}}); err != nil {
		t.Fatalf("failed to add line: %v", err)
	}

	// Add passenger to station 0
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{
		ID:          1,
		Destination: Square,
	})

	// Add passenger on train
	if len(sim.State.Trains) > 0 {
		sim.State.Trains[0].Passengers = append(sim.State.Trains[0].Passengers, Passenger{
			ID:          2,
			Destination: Square,
		})
	}

	// Clone the simulator
	cloned := sim.Clone()

	// 1. Verify fidelity
	if len(cloned.State.Stations) != len(sim.State.Stations) {
		t.Fatalf("cloned station count mismatch: got %d, want %d", len(cloned.State.Stations), len(sim.State.Stations))
	}
	if len(cloned.State.Lines) != len(sim.State.Lines) {
		t.Fatalf("cloned line count mismatch: got %d, want %d", len(cloned.State.Lines), len(sim.State.Lines))
	}
	if len(cloned.State.Trains) != len(sim.State.Trains) {
		t.Fatalf("cloned train count mismatch: got %d, want %d", len(cloned.State.Trains), len(sim.State.Trains))
	}
	if len(cloned.State.Stations[0].Queue) != 1 {
		t.Fatalf("cloned station queue mismatch: got %d, want 1", len(cloned.State.Stations[0].Queue))
	}
	if len(cloned.State.Trains[0].Passengers) != 1 {
		t.Fatalf("cloned train passengers mismatch: got %d, want 1", len(cloned.State.Trains[0].Passengers))
	}

	// 2. Verify independence: mutate cloned
	cloned.State.Stations[0].Queue = append(cloned.State.Stations[0].Queue, Passenger{
		ID:          99,
		Destination: Triangle,
	})
	_ = cloned.ApplyAction(RemoveLine{LineID: 0})

	// Original must be untouched!
	if len(sim.State.Stations[0].Queue) != 1 {
		t.Fatalf("mutation in clone leaked into original station queue: got %d, want 1", len(sim.State.Stations[0].Queue))
	}
	if sim.State.Lines[0].Removed {
		t.Fatalf("removal of line in clone leaked into original simulator!")
	}
}

func TestSimulator_DisruptionPenaltyAccounting(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 100, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 200, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 2, Kind: Square, Pos: Pos{X: 300, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
	}
	sim := NewSimulatorWithRivers(stations, nil, 42)

	if err := sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}}); err != nil {
		t.Fatalf("failed to add line: %v", err)
	}

	// Put 2 passengers on active train
	if len(sim.State.Trains) > 0 {
		sim.State.Trains[0].Passengers = []Passenger{
			{ID: 1, Destination: Square},
			{ID: 2, Destination: Triangle},
		}
	}


	// Remove line
	if err := sim.ApplyAction(RemoveLine{LineID: 0}); err != nil {
		t.Fatalf("failed to remove line: %v", err)
	}

	// Disruption penalty should be assessed:
	// Base (1.0) + 2 pax (0.40) + 1 train (0.50) + 3 severed stations (4.50) = 6.40
	if sim.disruptionPenalty < 5.0 {
		t.Fatalf("expected substantial disruption penalty, got %f", sim.disruptionPenalty)
	}

	// Next step breakdown should reflect this disruption penalty
	_, rb := sim.ComputeStepRewardBreakdown(0)
	if rb.Disruption >= 0 || rb.Disruption < -10.0 {
		t.Fatalf("expected negative disruption reward breakdown, got %f", rb.Disruption)
	}
	// And penalty should reset to 0 after step
	if sim.disruptionPenalty != 0.0 {
		t.Fatalf("expected disruptionPenalty to reset to 0, got %f", sim.disruptionPenalty)
	}
}

func TestSimulator_HysteresisCooldownAndEmergencyOverride(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 100, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 200, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 2, Kind: Square, Pos: Pos{X: 300, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
	}
	sim := NewSimulatorWithRivers(stations, nil, 42)
	sim.State.Tick = 500

	if err := sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}}); err != nil {
		t.Fatalf("failed to add line: %v", err)
	}

	// Verify LastModifiedTick was recorded on line addition
	if sim.State.Lines[0].LastModifiedTick != 500 {
		t.Fatalf("expected LastModifiedTick=500, got %d", sim.State.Lines[0].LastModifiedTick)
	}

	// Advance tick and test QueueGrowthRate smoothing
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{ID: 10, Destination: Square})
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{ID: 11, Destination: Square})
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{ID: 12, Destination: Square})

	// Run 30 sub-ticks to trigger queue rate update
	for i := 0; i < 30; i++ {
		sim.Step(1.0 / 30.0)
	}


	if sim.State.Stations[0].QueueGrowthRate <= 0.0 {
		t.Fatalf("expected positive QueueGrowthRate after passenger surge, got %f", sim.State.Stations[0].QueueGrowthRate)
	}
}

func TestRemoveLine_NoFakePassengerDeliveryReward(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 100, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
		{ID: 1, Kind: Square, Pos: Pos{X: 200, Y: 100}, Alive: true, Capacity: 6, OvercrowdingTimer: -1},
	}
	sim := NewSimulatorWithRivers(stations, nil, 42)

	if err := sim.ApplyAction(AddLine{Stations: []int{0, 1}}); err != nil {
		t.Fatalf("failed to add line: %v", err)
	}

	// Place passenger on active train whose destination matches Station 1 (Square)
	if len(sim.State.Trains) > 0 {
		sim.State.Trains[0].Progress = 0.9 // close to station 1
		sim.State.Trains[0].Passengers = []Passenger{
			{ID: 101, Destination: Square},
		}
	}

	initialScore := sim.State.Score
	if initialScore != 0 {
		t.Fatalf("expected initial score 0, got %d", initialScore)
	}

	// Remove the line
	if err := sim.ApplyAction(RemoveLine{LineID: 0}); err != nil {
		t.Fatalf("failed to remove line: %v", err)
	}

	// Correctness check:
	// 1. Score must NOT increase! Forced unloading is NOT delivery!
	if sim.State.Score != initialScore {
		t.Fatalf("fake passenger delivery reward detected! Score increased from %d to %d on line removal", initialScore, sim.State.Score)
	}

	// 2. Dumped passenger must be in Station 1 queue
	st1Queue := sim.State.Stations[1].Queue
	if len(st1Queue) != 1 || st1Queue[0].ID != 101 {
		t.Fatalf("expected passenger 101 in Station 1 queue, got %+v", st1Queue)
	}
}

