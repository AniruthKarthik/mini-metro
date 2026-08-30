package main

import (
	"math"
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

func TestTrainConditionalStationSlowdownAndPassThrough(t *testing.T) {
	// Create a loop line: Station 0 (Circle) -> Station 1 (Triangle) -> Station 2 (Square) -> Station 3 (Pentagon) -> (loop back to 0)
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Pentagon, Pos: engine.Pos{X: 30, Y: 0}},
	})

	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2, 3}})
	_ = sim.ApplyAction(engine.CloseLoop{LineID: 0})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	tr := &sim.State.Trains[0]
	tr.Segment = 1  // departing Station 1 toward Station 2
	tr.Progress = 0.5
	tr.JustArrived = false
	tr.JustDeparted = false

	// Station 2 has NO queued passengers and train has NO onboard passengers.
	// Train should NOT stop at Station 2 when it reaches progress >= 1.0.
	for i := 0; i < 50; i++ {
		sim.Step(0.01)
		if tr.Segment == 2 {
			break
		}
	}

	if tr.Segment != 2 {
		t.Fatalf("expected train to reach Station 2")
	}
	if tr.JustArrived {
		t.Errorf("expected train to pass through Station 2 without stopping when no passenger service is needed")
	}

	// Now seed Station 3 with a passenger wanting Circle (Station 0)
	sim.State.Stations[3].Queue = append(sim.State.Stations[3].Queue, engine.Passenger{
		Destination: engine.Circle,
	})

	// Advance train toward Station 3
	for i := 0; i < 120; i++ {
		sim.Step(0.01)
		if tr.Segment == 3 && tr.JustArrived {
			break
		}
	}

	if tr.Segment != 3 || !tr.JustArrived {
		t.Errorf("expected train to stop at Station 3 where passenger exchange is needed")
	}
}

func TestSymmetricAccelerationAndDeceleration(t *testing.T) {
	// Verify exact numerical symmetry of acceleration leaving station (p=0..0.25)
	// and deceleration entering station (p=0.75..1.0), and constant speed (1.0) along intermediate path (0.25..0.75).
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 100, Y: 0}},
	})
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.AddTrain{LineID: 0})

	tr := &sim.State.Trains[0]

	// At start of segment (p=0.0): departing station 0 (accelFromStart=true)
	tr.Progress = 0.0
	tr.JustDeparted = true

	// Test symmetric points: p=0.0 vs p=1.0, p=0.10 vs p=0.90, p=0.20 vs p=0.80
	// Intermediate path points: p=0.30, p=0.50, p=0.70 must have constant max speed factor (1.0)
	testCases := []struct {
		accelP float64
		decelP float64
	}{
		{0.01, 0.99},
		{0.05, 0.95},
		{0.10, 0.90},
		{0.15, 0.85},
		{0.20, 0.80},
		{0.24, 0.76},
	}

	for _, tc := range testCases {
		// Calculate step delta at accelP (leaving station 0)
		tr.Progress = tc.accelP
		tr.JustArrived = false
		tr.JustDeparted = true
		pAccelStart := tr.Progress
		sim.Step(0.001)
		accelDelta := tr.Progress - pAccelStart

		// Calculate step delta at decelP (entering terminal station 1)
		tr.Progress = tc.decelP
		tr.JustArrived = false
		tr.JustDeparted = false
		pDecelStart := tr.Progress
		sim.Step(0.001)
		decelDelta := tr.Progress - pDecelStart

		if math.Abs(accelDelta-decelDelta) > 1e-6 {
			t.Errorf("asymmetry detected between accel at p=%.2f (delta=%.6f) and decel at p=%.2f (delta=%.6f)",
				tc.accelP, accelDelta, tc.decelP, decelDelta)
		}
	}

	// Verify constant cruising speed across intermediate path (p=0.25 to p=0.75)
	tr.Progress = 0.30
	tr.JustDeparted = false
	p1Start := tr.Progress
	sim.Step(0.001)
	delta30 := tr.Progress - p1Start

	tr.Progress = 0.50
	tr.JustDeparted = false
	p2Start := tr.Progress
	sim.Step(0.001)
	delta50 := tr.Progress - p2Start

	if math.Abs(delta30-delta50) > 1e-6 {
		t.Errorf("intermediate speed was not constant between p=0.30 (delta=%.6f) and p=0.50 (delta=%.6f)", delta30, delta50)
	}
}
