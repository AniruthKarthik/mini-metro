package engine

import (
	"testing"
)

func createTestStations() []Station {
	return []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true},
		{ID: 3, Kind: Star, Pos: Pos{X: 30, Y: 0}, Alive: true},
		{ID: 4, Kind: Pentagon, Pos: Pos{X: 40, Y: 0}, Alive: true},
	}
}

func TestNewSimulator(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	if !sim.State.Alive {
		t.Errorf("Expected GameState.Alive to be true upon initialization")
	}
	if len(sim.State.Stations) != 5 {
		t.Errorf("Expected 5 stations, got %d", len(sim.State.Stations))
	}
}

func TestAddLineValidation(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	// Valid line
	sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	if len(sim.State.Lines) != 1 {
		t.Fatalf("Expected 1 line, got %d", len(sim.State.Lines))
	}

	// Line with less than 2 stations (invalid)
	sim.ApplyAction(AddLine{Stations: []int{2}})
	if len(sim.State.Lines) != 1 {
		t.Errorf("Expected 1 line after invalid AddLine, got %d", len(sim.State.Lines))
	}

	// Line with invalid station ID (out of bounds)
	sim.ApplyAction(AddLine{Stations: []int{0, 999}})
	if len(sim.State.Lines) != 1 {
		t.Errorf("Expected 1 line after out-of-bounds AddLine, got %d", len(sim.State.Lines))
	}
}

func TestTrainBoardingAndAlighting(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	// Add Line 0: Station 0 (Circle) -> Station 1 (Triangle)
	sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	sim.ApplyAction(AddTrain{LineID: 0})

	if len(sim.State.Trains) != 1 {
		t.Fatalf("Expected 1 train, got %d", len(sim.State.Trains))
	}

	// Manually place a Triangle passenger in Station 0 queue
	sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{
		Origin:      0,
		Destination: Triangle,
		SpawnTick:   0,
	})

	// Step 1: Train is at Station 0 and JustArrived is true, so it should board the passenger
	sim.Step(0.1)

	tr := &sim.State.Trains[0]
	if len(tr.Passengers) != 1 || tr.Passengers[0].Destination != Triangle {
		t.Errorf("Expected train to board Triangle passenger at Station 0, got passengers %v", tr.Passengers)
	}
	if len(sim.State.Stations[0].Queue) != 0 {
		t.Errorf("Expected Station 0 queue to be empty after boarding, got %d", len(sim.State.Stations[0].Queue))
	}

	// Step until train reaches Station 1
	for i := 0; i < 25; i++ {
		sim.Step(0.1)
		if sim.State.Score > 0 {
			break
		}
	}

	if sim.State.Score != 1 {
		t.Errorf("Expected score to be 1 after passenger alights at Triangle station, got %d", sim.State.Score)
	}
	if len(sim.State.Trains[0].Passengers) != 0 {
		t.Errorf("Expected train to have 0 passengers after alighting, got %d", len(sim.State.Trains[0].Passengers))
	}
}

func TestExtendAndRemoveLine(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	sim.ApplyAction(AddTrain{LineID: 0})

	// Extend line to station 2
	sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})
	if len(sim.State.Lines[0].Stations) != 3 {
		t.Errorf("Expected line 0 to have 3 stations, got %d", len(sim.State.Lines[0].Stations))
	}

	// Remove line
	sim.ApplyAction(RemoveLine{LineID: 0})
	if !sim.State.Lines[0].Removed {
		t.Errorf("Expected line 0 to be marked as Removed")
	}
	if sim.State.Trains[0].Active {
		t.Errorf("Expected train on removed line to be deactivated")
	}
}

func TestGameOverTrigger(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	// Fill queue of Station 0 beyond maxQsize (20)
	for i := 0; i < 25; i++ {
		sim.State.Stations[0].Queue = append(sim.State.Stations[0].Queue, Passenger{
			Origin:      0,
			Destination: Triangle,
			SpawnTick:   0,
		})
	}

	sim.Step(0.1)

	if sim.State.Alive {
		t.Errorf("Expected GameState.Alive to be false when queue size > 20")
	}

	// Further steps should be no-ops
	currentTick := sim.State.Tick
	sim.Step(0.1)
	if sim.State.Tick != currentTick {
		t.Errorf("Expected Tick count to remain unchanged after game over")
	}
}

func TestObservation(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	sim.ApplyAction(AddTrain{LineID: 0})

	obs := sim.Observation()
	if len(obs.StationKinds) != 5 {
		t.Errorf("Expected 5 station kinds in observation, got %d", len(obs.StationKinds))
	}
	if len(obs.TrainLineIDs) != 1 {
		t.Errorf("Expected 1 active train in observation, got %d", len(obs.TrainLineIDs))
	}
}

func TestSafetyChecksAgainstInvalidActions(t *testing.T) {
	stations := createTestStations()
	sim := NewSimulator(stations)

	// These should not panic
	sim.ApplyAction(ExtendLine{LineID: 99, StationID: 0})
	sim.ApplyAction(ExtendLine{LineID: 0, StationID: 99})
	sim.ApplyAction(AddTrain{LineID: 99})
	sim.ApplyAction(RemoveLine{LineID: 99})

	sim.Step(1.0)
}
