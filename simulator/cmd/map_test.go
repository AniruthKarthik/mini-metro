package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestCityMapPresets(t *testing.T) {
	// London Map
	simLondon := engine.NewSimulatorWithMap(engine.LondonMap())
	if simLondon.State.MapName != "London" {
		t.Errorf("expected London map name, got %s", simLondon.State.MapName)
	}
	if len(simLondon.State.Rivers) == 0 {
		t.Errorf("expected London Thames river geometry")
	}

	// NYC Map
	simNYC := engine.NewSimulatorWithMap(engine.NYCMap())
	if simNYC.State.MapName != "New York City" {
		t.Errorf("expected NYC map name, got %s", simNYC.State.MapName)
	}
	if len(simNYC.State.Rivers) != 2 {
		t.Errorf("expected 2 river channels for NYC (Hudson & East Rivers), got %d", len(simNYC.State.Rivers))
	}
	if len(simNYC.State.WaterPolygons) != 1 {
		t.Errorf("expected Upper New York Bay polygon in NYC map")
	}

	// Tokyo Map
	simTokyo := engine.NewSimulatorWithMap(engine.TokyoMap())
	if simTokyo.State.MapName != "Tokyo" {
		t.Errorf("expected Tokyo map name, got %s", simTokyo.State.MapName)
	}
	if len(simTokyo.State.Rivers) == 0 {
		t.Errorf("expected Sumida River in Tokyo map")
	}
	if len(simTokyo.State.WaterPolygons) == 0 {
		t.Errorf("expected Tokyo Bay polygon in Tokyo map")
	}

	// Berlin Map
	simBerlin := engine.NewSimulatorWithMap(engine.BerlinMap())
	if simBerlin.State.MapName != "Berlin" {
		t.Errorf("expected Berlin map name, got %s", simBerlin.State.MapName)
	}
	if len(simBerlin.State.Rivers) != 0 {
		t.Errorf("expected 0 rivers for Berlin, got %d", len(simBerlin.State.Rivers))
	}
	if len(simBerlin.State.WaterPolygons) != 0 {
		t.Errorf("expected 0 water polygons for Berlin, got %d", len(simBerlin.State.WaterPolygons))
	}
	if simBerlin.State.Resources.Tunnels != 0 {
		t.Errorf("expected 0 initial tunnels for Berlin, got %d", simBerlin.State.Resources.Tunnels)
	}
	if simBerlin.State.Resources.Lines != 3 {
		t.Errorf("expected 3 initial lines for Berlin, got %d", simBerlin.State.Resources.Lines)
	}
	if simBerlin.State.Resources.Trains != 3 {
		t.Errorf("expected 3 initial trains for Berlin, got %d", simBerlin.State.Resources.Trains)
	}
	if len(simBerlin.State.Stations) != 3 {
		t.Fatalf("expected 3 initial stations for Berlin, got %d", len(simBerlin.State.Stations))
	}
	// Verify that initial stations strictly preserve canonical kinds in order:
	if simBerlin.State.Stations[0].Kind != engine.Circle {
		t.Errorf("expected station 0 to be Circle, got %v", simBerlin.State.Stations[0].Kind)
	}
	if simBerlin.State.Stations[1].Kind != engine.Triangle {
		t.Errorf("expected station 1 to be Triangle, got %v", simBerlin.State.Stations[1].Kind)
	}
	if simBerlin.State.Stations[2].Kind != engine.Square {
		t.Errorf("expected station 2 to be Square, got %v", simBerlin.State.Stations[2].Kind)
	}

	// Verify that different seeds produce different randomized station positions
	sim1 := engine.NewSimulatorWithMap(engine.BerlinMap(), 101)
	sim2 := engine.NewSimulatorWithMap(engine.BerlinMap(), 202)
	if sim1.State.Stations[0].Pos == sim2.State.Stations[0].Pos && sim1.State.Stations[1].Pos == sim2.State.Stations[1].Pos {
		t.Errorf("expected different seeds to produce different randomized initial station positions")
	}
}

func TestLondonMapThamesCrossing(t *testing.T) {
	simLondon := engine.NewSimulatorWithMap(engine.LondonMap())
	// Place stations 0 and 1 on opposite sides of Thames to test crossing tunnel consumption
	simLondon.State.Stations[0].Pos = engine.Pos{X: 20, Y: 25}
	simLondon.State.Stations[1].Pos = engine.Pos{X: 50, Y: 60}

	// Add line connecting station 0 (20,25) to station 1 (50,60) which crosses Thames
	err := simLondon.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("expected AddLine across Thames to succeed using London's initial tunnel budget, got: %v", err)
	}
	if simLondon.State.Resources.Tunnels != 2 {
		t.Errorf("expected one London tunnel token to be spent on Thames river crossing, got %d", simLondon.State.Resources.Tunnels)
	}
}

func TestBerlinMapNoTunnelsRequired(t *testing.T) {
	simBerlin := engine.NewSimulatorWithMap(engine.BerlinMap())
	simBerlin.State.Stations[0].Pos = engine.Pos{X: 20, Y: 25}
	simBerlin.State.Stations[1].Pos = engine.Pos{X: 50, Y: 60}

	// In London, connecting station 0 (20,25) to station 1 (50,60) requires a tunnel.
	// In Berlin, there is no water, so it must succeed with 0 tunnels in the resource pool.
	if simBerlin.State.Resources.Tunnels != 0 {
		t.Fatalf("expected 0 tunnels initially in Berlin, got %d", simBerlin.State.Resources.Tunnels)
	}

	err := simBerlin.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("expected AddLine across identical London coordinates to succeed in Berlin with no tunnels, got: %v", err)
	}
	if simBerlin.State.Resources.Tunnels != 0 {
		t.Errorf("expected tunnel count to remain 0 in Berlin, got %d", simBerlin.State.Resources.Tunnels)
	}

	// Verify that weekly reward generation on Berlin never produces RewardTunnel
	for tick := uint64(0); tick < 5000; tick++ {
		simBerlin.Step(1.0 / 30.0)
		for _, choice := range simBerlin.State.PendingRewardChoices {
			if choice == engine.RewardTunnel {
				t.Fatalf("unexpected RewardTunnel offered in Berlin weekly rewards: %+v", simBerlin.State.PendingRewardChoices)
			}
		}
	}
}
