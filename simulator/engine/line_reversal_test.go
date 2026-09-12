package engine

import (
	"testing"
)

func TestReverseLineStationsAndTunnels(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)
	err := sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("failed to add line: %v", err)
	}
	err = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2, FromFront: false})
	if err != nil {
		t.Fatalf("failed to extend line: %v", err)
	}

	line := &sim.State.Lines[0]
	if len(line.Stations) != 3 {
		t.Fatalf("expected 3 stations on line 0, got %d", len(line.Stations))
	}

	// Set specific tunnel flags for testing
	line.TunnelAt = []bool{false, true}

	// Reverse line
	if err := sim.ReverseLine(0); err != nil {
		t.Fatalf("ReverseLine failed: %v", err)
	}

	// Check reversed station sequence
	expectedStations := []int{2, 1, 0}
	for i, st := range line.Stations {
		if st != expectedStations[i] {
			t.Fatalf("expected station[%d] == %d, got %d", i, expectedStations[i], st)
		}
	}

	// Check reversed tunnel flags
	expectedTunnels := []bool{true, false}
	for i, tun := range line.TunnelAt {
		if tun != expectedTunnels[i] {
			t.Fatalf("expected tunnel[%d] == %v, got %v", i, expectedTunnels[i], tun)
		}
	}

	// Reverse back: must restore original state
	if err := sim.ReverseLine(0); err != nil {
		t.Fatalf("ReverseLine back failed: %v", err)
	}
	for i, st := range line.Stations {
		if st != i {
			t.Fatalf("expected restored station[%d] == %d, got %d", i, i, st)
		}
	}
	if line.TunnelAt[0] != false || line.TunnelAt[1] != true {
		t.Fatalf("expected restored tunnels [false, true], got %v", line.TunnelAt)
	}
}

func TestReverseLineActiveTrains(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2, FromFront: false})

	// Line has stations [0, 1, 2]
	// Configure Train 0: at segment 0 (station 0), direction +1
	sim.State.Trains[0].Segment = 0
	sim.State.Trains[0].Direction = 1

	// Add Train 1: at segment 2 (station 2), direction -1
	sim.State.Trains = append(sim.State.Trains, Train{
		ID:        1,
		LineID:    0,
		Segment:   2,
		Direction: -1,
		Capacity:  6,
		Carriages: 1,
		Active:    true,
	})

	if err := sim.ReverseLine(0); err != nil {
		t.Fatalf("ReverseLine failed: %v", err)
	}

	// Reversed line has stations [2, 1, 0]
	// Train 0: was segment 0 (station 0), now segment (3-1)-0 = 2 (station 0), direction -1 (towards station 1)
	tr0 := sim.State.Trains[0]
	if tr0.Segment != 2 {
		t.Fatalf("expected Train 0 segment == 2, got %d", tr0.Segment)
	}
	if tr0.Direction != -1 {
		t.Fatalf("expected Train 0 direction == -1, got %d", tr0.Direction)
	}
	if sim.State.Lines[0].Stations[tr0.Segment] != 0 {
		t.Fatalf("expected Train 0 to be at station 0, got %d", sim.State.Lines[0].Stations[tr0.Segment])
	}

	// Train 1: was segment 2 (station 2), now segment (3-1)-2 = 0 (station 2), direction +1 (towards station 1)
	tr1 := sim.State.Trains[1]
	if tr1.Segment != 0 {
		t.Fatalf("expected Train 1 segment == 0, got %d", tr1.Segment)
	}
	if tr1.Direction != 1 {
		t.Fatalf("expected Train 1 direction == 1, got %d", tr1.Direction)
	}
	if sim.State.Lines[0].Stations[tr1.Segment] != 2 {
		t.Fatalf("expected Train 1 to be at station 2, got %d", sim.State.Lines[0].Stations[tr1.Segment])
	}
}

func TestReverseLineKinematicContinuity(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)
	sim.State.Stations = append(sim.State.Stations, Station{
		ID: 3, Kind: Cross, Pos: Pos{X: 90, Y: 70}, Alive: true, Capacity: 6,
	})
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2, FromFront: false})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 3, FromFront: false})

	// Line has stations [0, 1, 2, 3]
	testCases := []struct {
		seg  int
		dir  int
		prog float64
	}{
		{seg: 0, dir: 1, prog: 0.0},
		{seg: 0, dir: 1, prog: 0.25},
		{seg: 0, dir: 1, prog: 0.8},
		{seg: 1, dir: 1, prog: 0.5},
		{seg: 2, dir: -1, prog: 0.1},
		{seg: 2, dir: -1, prog: 0.75},
		{seg: 3, dir: -1, prog: 0.0},
		{seg: 3, dir: -1, prog: 0.99},
	}

	getTrainSpatialState := func(s *Simulator, tr *Train) (x, y float64, fromSt, toSt int) {
		line := &s.State.Lines[tr.LineID]
		fromSt = line.Stations[tr.Segment]
		nextIdx := tr.Segment + tr.Direction
		if nextIdx < 0 {
			nextIdx = 0
		}
		if nextIdx >= len(line.Stations) {
			nextIdx = len(line.Stations) - 1
		}
		toSt = line.Stations[nextIdx]
		stA := &s.State.Stations[fromSt]
		stB := &s.State.Stations[toSt]
		x = float64(stA.Pos.X) + tr.Progress*float64(stB.Pos.X-stA.Pos.X)
		y = float64(stA.Pos.Y) + tr.Progress*float64(stB.Pos.Y-stA.Pos.Y)
		return
	}

	for idx, tc := range testCases {
		sim.State.Trains[0].Segment = tc.seg
		sim.State.Trains[0].Direction = tc.dir
		sim.State.Trains[0].Progress = tc.prog

		xBefore, yBefore, fromBefore, toBefore := getTrainSpatialState(sim, &sim.State.Trains[0])

		if err := sim.ReverseLine(0); err != nil {
			t.Fatalf("case %d: ReverseLine failed: %v", idx, err)
		}

		xAfter, yAfter, fromAfter, toAfter := getTrainSpatialState(sim, &sim.State.Trains[0])

		diffX := xBefore - xAfter
		diffY := yBefore - yAfter
		if diffX < 0 {
			diffX = -diffX
		}
		if diffY < 0 {
			diffY = -diffY
		}

		if diffX > 1e-4 || diffY > 1e-4 {
			t.Errorf("case %d: spatial jump detected: before=(%.4f, %.4f), after=(%.4f, %.4f), diff=(%.4f, %.4f)",
				idx, xBefore, yBefore, xAfter, yAfter, diffX, diffY)
		}

		if fromBefore != fromAfter || toBefore != toAfter {
			t.Errorf("case %d: kinematic heading changed: before=(from %d to %d), after=(from %d to %d)",
				idx, fromBefore, toBefore, fromAfter, toAfter)
		}

		// Reverse back to original order for next test case
		_ = sim.ReverseLine(0)
	}
}

func TestReverseLineLoopIgnored(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2, FromFront: false})

	sim.State.Lines[0].IsLoop = true
	origStations := append([]int(nil), sim.State.Lines[0].Stations...)

	if err := sim.ReverseLine(0); err != nil {
		t.Fatalf("ReverseLine on loop should return nil, got %v", err)
	}

	for i, st := range sim.State.Lines[0].Stations {
		if st != origStations[i] {
			t.Fatalf("expected loop station[%d] == %d unchanged, got %d", i, origStations[i], st)
		}
	}
}


