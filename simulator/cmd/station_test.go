package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestExpandedStationKinds(t *testing.T) {
	kinds := []engine.StationKind{
		engine.Circle, engine.Triangle, engine.Square, engine.Star, engine.Pentagon,
		engine.Gem, engine.Sector, engine.Cross, engine.Drop, engine.Oval,
	}

	stations := make([]engine.Station, len(kinds))
	for i, k := range kinds {
		stations[i] = engine.Station{
			ID:   i,
			Kind: k,
			Pos:  engine.Pos{X: float64(i * 10), Y: 0},
		}
	}

	sim := engine.NewSimulator(stations)
	for i, k := range kinds {
		if sim.State.Stations[i].Kind != k {
			t.Errorf("station %d: expected kind %s, got %s", i, k, sim.State.Stations[i].Kind)
		}
	}
}

func TestInterchangeHubUpgrade(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})

	sim.State.Resources.Grant(engine.RewardInterchange)

	err := sim.ApplyAction(engine.UpgradeInterchange{StationID: 0})
	if err != nil {
		t.Fatalf("UpgradeInterchange failed: %v", err)
	}

	st := &sim.State.Stations[0]
	if !st.IsInterchange {
		t.Errorf("expected station 0 to be marked as interchange hub")
	}
	if st.Capacity != 18 {
		t.Errorf("expected interchange capacity = 18, got %d", st.Capacity)
	}
}
