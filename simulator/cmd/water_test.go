package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestWaterCrossingsAndTunnelGating(t *testing.T) {
	rivers := []engine.RiverSegment{
		{From: engine.Pos{X: 15, Y: 0}, To: engine.Pos{X: 15, Y: 100}, Width: 4.0},
	}
	sim := engine.NewSimulatorWithRivers([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 50}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 30, Y: 50}},
	}, rivers)

	// Tunnel pool is empty initially
	err := sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err == nil {
		t.Errorf("expected AddLine across river to fail without tunnel tokens")
	}

	// Grant tunnel token
	sim.State.Resources.Grant(engine.RewardTunnel)

	err = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("expected AddLine across river to succeed after granting tunnel token, got: %v", err)
	}

	if sim.State.Resources.Tunnels != 0 {
		t.Errorf("expected 0 tunnel tokens remaining after spending, got %d", sim.State.Resources.Tunnels)
	}
	if !sim.State.Lines[0].TunnelAt[0] {
		t.Errorf("expected segment 0 to be marked as tunnel crossing")
	}
}

func TestWaterPolygonCrossing(t *testing.T) {
	polygons := []engine.WaterPolygon{
		{Vertices: []engine.Pos{
			{X: 10, Y: 10}, {X: 40, Y: 10}, {X: 40, Y: 40}, {X: 10, Y: 40},
		}},
	}
	sim := engine.NewSimulatorWithWater([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 20}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 50, Y: 20}},
	}, nil, polygons)

	// Attempt AddLine across 2D water polygon without tunnel token
	err := sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err == nil {
		t.Errorf("expected AddLine across water polygon to fail without tunnel tokens")
	}
}
