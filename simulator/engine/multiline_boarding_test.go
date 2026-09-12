package engine

import (
	"testing"
)

func TestMultiLineBoardingEquality(t *testing.T) {
	// Setup a station 0 (Circle) served by two lines:
	// Line 0 connects 0 -> 1 (Triangle)
	// Line 1 connects 0 -> 2 (Triangle)
	sim := NewSimulatorWithMap(MapConfig{
		Name:             "TestMultiLine",
		MaxLines:         3,
		MaxTrainsPerLine: 2,
		InitialResources: ResourcePool{Lines: 2, Trains: 2},
		InitialStations: []Station{
			{ID: 0, Kind: Circle, Pos: Pos{X: 10, Y: 10}},
			{ID: 1, Kind: Triangle, Pos: Pos{X: 20, Y: 10}},
			{ID: 2, Kind: Triangle, Pos: Pos{X: 10, Y: 20.1}},
		},
	})

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 2}})
	sim.rebuildGraphIfNeeded()

	// Place 2 passengers at Station 0 wanting Triangle
	sim.State.Stations[0].Queue = []Passenger{
		{ID: 101, Destination: Triangle, SpawnTick: 0},
		{ID: 102, Destination: Triangle, SpawnTick: 0},
	}

	// Train on Line 1 arrives at Station 0
	tr1 := &sim.State.Trains[1]
	tr1.Segment = 0
	tr1.Direction = 1
	tr1.Active = true
	tr1.JustArrived = true
	tr1.DwellRemaining = 0
	tr1.ServiceElapsed = 0

	st := &sim.State.Stations[0]

	if !sim.hasServiceWork(tr1, st, 0) {
		t.Fatalf("Expected train on Line 1 to have service work at Station 0!")
	}

	boarded := sim.serviceOnePassenger(tr1, st, 0)
	if !boarded {
		t.Fatalf("Expected train on Line 1 to board passenger at Station 0!")
	}
	if len(tr1.Passengers) != 1 || tr1.Passengers[0].ID != 101 {
		t.Fatalf("Expected passenger 101 to board train 1, got %v", tr1.Passengers)
	}
	if len(st.Queue) != 1 || st.Queue[0].ID != 102 {
		t.Fatalf("Expected remaining queue to be [102], got %v", st.Queue)
	}

	// Now train on Line 0 arrives at Station 0
	tr0 := &sim.State.Trains[0]
	tr0.Segment = 0
	tr0.Direction = 1
	tr0.Active = true
	tr0.JustArrived = true
	tr0.DwellRemaining = 0
	tr0.ServiceElapsed = 0

	if !sim.hasServiceWork(tr0, st, 0) {
		t.Fatalf("Expected train on Line 0 to have service work at Station 0!")
	}

	boarded2 := sim.serviceOnePassenger(tr0, st, 0)
	if !boarded2 {
		t.Fatalf("Expected train on Line 0 to board passenger 102!")
	}
	if len(tr0.Passengers) != 1 || tr0.Passengers[0].ID != 102 {
		t.Fatalf("Expected passenger 102 to board train 0, got %v", tr0.Passengers)
	}
	if len(st.Queue) != 0 {
		t.Fatalf("Expected queue to be empty, got %d", len(st.Queue))
	}
}
