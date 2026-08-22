package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestAcceleratingPassengerSpawnRate(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
	})

	initialRate := sim.CurrentSpawnRate()
	if initialRate <= 0.0 {
		t.Errorf("expected initial spawn rate > 0, got %f", initialRate)
	}

	for i := 0; i < 1000; i++ {
		sim.Step(0.01)
	}

	acceleratedRate := sim.CurrentSpawnRate()
	if acceleratedRate <= initialRate {
		t.Errorf("expected spawn rate to accelerate over ticks, initial=%f new=%f", initialRate, acceleratedRate)
	}
}

func TestWeightedPassengerDestinationSampling(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Star, Pos: engine.Pos{X: 20, Y: 0}},
	})

	rareDestinationSpawned := false
	activeFilteredOnly := true

	for i := 0; i < 4000; i++ {
		if len(sim.State.PendingRewardChoices) > 0 {
			if err := sim.ApplyAction(engine.ChooseReward{Choice: sim.State.PendingRewardChoices[0]}); err != nil {
				t.Fatalf("choose reward failed: %v", err)
			}
		}
		sim.Step(0.05)
		for _, st := range sim.State.Stations {
			for _, p := range st.Queue {
				if p.Destination == engine.Star {
					rareDestinationSpawned = true
				}
				foundActiveKind := false
				for _, s := range sim.State.Stations {
					if s.Alive && s.Kind == p.Destination {
						foundActiveKind = true
						break
					}
				}
				if !foundActiveKind {
					activeFilteredOnly = false
				}
			}
		}
	}

	if !rareDestinationSpawned {
		t.Errorf("expected passengers to spawn targeting rare active station (Star)")
	}
	if !activeFilteredOnly {
		t.Errorf("expected passengers to target only active station kinds present on the map")
	}
}
