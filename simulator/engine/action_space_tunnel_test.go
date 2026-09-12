package engine

import (
	"testing"
)

func TestInsertStationNetTunnelsMask(t *testing.T) {
	// Create a simulator on London map
	sim := NewSimulatorWithMap(LondonMap())

	if len(sim.State.Stations) < 3 {
		t.Fatalf("expected at least 3 stations on map")
	}

	// Create a river across the center at Y=50
	sim.State.Rivers = []RiverSegment{
		{From: Pos{X: 0, Y: 50}, To: Pos{X: 100, Y: 50}, Width: 4.0},
	}
	sim.State.WaterPolygons = nil

	// Station 0 at (20, 20) - south of river
	// Station 1 at (80, 20) - south of river (segment 0-1 does NOT cross river)
	// Station 2 at (50, 80) - north of river (inserting 2 creates segments 0-2 and 2-1, BOTH crossing river!)
	sim.State.Stations[0].Pos = Pos{X: 20, Y: 20}
	sim.State.Stations[0].Alive = true
	sim.State.Stations[1].Pos = Pos{X: 80, Y: 20}
	sim.State.Stations[1].Alive = true
	sim.State.Stations[2].Pos = Pos{X: 50, Y: 80}
	sim.State.Stations[2].Alive = true

	// Add Line 0 connecting station 0 and station 1 (0 tunnels needed)
	sim.State.Resources.Lines = 1
	sim.State.Resources.Trains = 1
	sim.State.Resources.Tunnels = 1 // Only 1 tunnel token in inventory!

	err := sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("failed to add line: %v", err)
	}

	// Verify segment 0-1 does not cross water
	if CrossesWater(sim.State.Stations[0].Pos, sim.State.Stations[1].Pos, sim.State.Rivers, sim.State.WaterPolygons) {
		t.Fatalf("segment 0-1 should not cross water")
	}

	// Verify segments 0-2 and 2-1 cross water (2 tunnels needed)
	if !CrossesWater(sim.State.Stations[0].Pos, sim.State.Stations[2].Pos, sim.State.Rivers, sim.State.WaterPolygons) {
		t.Fatalf("segment 0-2 must cross water")
	}
	if !CrossesWater(sim.State.Stations[2].Pos, sim.State.Stations[1].Pos, sim.State.Rivers, sim.State.WaterPolygons) {
		t.Fatalf("segment 2-1 must cross water")
	}

	// Insert station 2 into line 0 at index 1: requires 2 net tunnels
	mask := make([]bool, MaxActionSpaceSize())
	sim.GetActionMask(mask)

	// Action ID for InsertStation(line=0, station=2, index=1):
	// idx = (0 * 30 + 2) * 15 + (1 - 1) = 30
	actionID := InsertStationOffset + 30
	act, err := ActionFromIndex(actionID)
	if err != nil {
		t.Fatalf("failed to decode action %d: %v", actionID, err)
	}
	ins, ok := act.(InsertStation)
	if !ok || ins.LineID != 0 || ins.StationID != 2 || ins.Index != 1 {
		t.Fatalf("unexpected decoded action: %+v", act)
	}

	// With only 1 tunnel token, mask MUST be false
	if mask[actionID] {
		t.Errorf("expected action %d to be MASKED (false) because 2 tunnels needed but only 1 available", actionID)
	}

	// Grant 1 more tunnel (total 2)
	sim.State.Resources.Tunnels = 2
	sim.GetActionMask(mask)

	// With 2 tunnel tokens, mask MUST be true
	if !mask[actionID] {
		t.Errorf("expected action %d to be UNMASKED (true) when 2 tunnels are available", actionID)
	}

	// Apply action to ensure simulator accepts it with 2 tunnels
	err = sim.ApplyAction(ins)
	if err != nil {
		t.Fatalf("failed to apply insert station with 2 tunnels: %v", err)
	}
	if sim.State.Resources.Tunnels != 0 {
		t.Fatalf("expected 0 tunnels remaining after spending 2, got %d", sim.State.Resources.Tunnels)
	}
}
