package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestTrainPhysicsAndMovement(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	sim.Step(0.01)

	tr := &sim.State.Trains[0]
	if tr.Progress <= 0 {
		t.Errorf("expected train to advance progress on Step, got %f", tr.Progress)
	}
}

func TestTrainRepositioningAction(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	err := sim.ApplyAction(engine.RepositionTrain{
		TrainID:   0,
		Segment:   2,
		Direction: -1,
	})
	if err != nil {
		t.Fatalf("RepositionTrain failed: %v", err)
	}

	tr := &sim.State.Trains[0]
	if tr.Segment != 2 {
		t.Errorf("expected train segment = 2, got %d", tr.Segment)
	}
	if tr.Direction != -1 {
		t.Errorf("expected train direction = -1, got %d", tr.Direction)
	}
}
