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
}

func TestLondonMapThamesCrossing(t *testing.T) {
	simLondon := engine.NewSimulatorWithMap(engine.LondonMap())

	// Add line connecting station 0 (20,25) to station 1 (50,60) which crosses Thames
	err := simLondon.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("expected AddLine across Thames to succeed using London's initial tunnel token, got: %v", err)
	}
	if simLondon.State.Resources.Tunnels != 0 {
		t.Errorf("expected London tunnel token to be spent on Thames river crossing")
	}
}
