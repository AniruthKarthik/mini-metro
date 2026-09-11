package engine

import (
	"testing"
)

func TestRemoveLine_ResourceRefunds(t *testing.T) {
	// Water between station 0 and 1 requiring a tunnel
	rivers := []RiverSegment{
		{From: Pos{X: 5, Y: -10}, To: Pos{X: 5, Y: 10}},
	}
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	}
	sim := NewSimulatorWithRivers(stations, rivers, 42)

	// Grant resources for tunnel and carriage
	sim.State.Resources.Grant(RewardTunnel)
	sim.State.Resources.Grant(RewardCarriage)

	initLines := sim.State.Resources.Lines
	initTrains := sim.State.Resources.Trains
	initTunnels := sim.State.Resources.Tunnels
	initCarriages := sim.State.Resources.Carriages

	// Add Line 0 (stations 0 and 1 across river -> spends 1 line, 1 train, 1 tunnel)
	err := sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("unexpected error adding line: %v", err)
	}

	// Add carriage to line 0 (spends 1 carriage)
	err = sim.ApplyAction(AddCarriage{LineID: 0})
	if err != nil {
		t.Fatalf("unexpected error adding carriage: %v", err)
	}

	if sim.State.Resources.Lines != initLines-1 {
		t.Errorf("expected lines %d, got %d", initLines-1, sim.State.Resources.Lines)
	}
	if sim.State.Resources.Trains != initTrains-1 {
		t.Errorf("expected trains %d, got %d", initTrains-1, sim.State.Resources.Trains)
	}
	if sim.State.Resources.Tunnels != initTunnels-1 {
		t.Errorf("expected tunnels %d, got %d", initTunnels-1, sim.State.Resources.Tunnels)
	}
	if sim.State.Resources.Carriages != initCarriages-1 {
		t.Errorf("expected carriages %d, got %d", initCarriages-1, sim.State.Resources.Carriages)
	}

	// Remove Line 0 -> should refund 1 line, 1 train, 1 carriage, 1 tunnel
	err = sim.ApplyAction(RemoveLine{LineID: 0})
	if err != nil {
		t.Fatalf("unexpected error removing line: %v", err)
	}

	if sim.State.Resources.Lines != initLines {
		t.Errorf("expected refunded lines %d, got %d", initLines, sim.State.Resources.Lines)
	}
	if sim.State.Resources.Trains != initTrains {
		t.Errorf("expected refunded trains %d, got %d", initTrains, sim.State.Resources.Trains)
	}
	if sim.State.Resources.Tunnels != initTunnels {
		t.Errorf("expected refunded tunnels %d, got %d", initTunnels, sim.State.Resources.Tunnels)
	}
	if sim.State.Resources.Carriages != initCarriages {
		t.Errorf("expected refunded carriages %d, got %d", initCarriages, sim.State.Resources.Carriages)
	}

	if !sim.State.Lines[0].Removed {
		t.Errorf("expected line 0 to be marked removed")
	}
	if sim.State.Trains[0].Active {
		t.Errorf("expected train 0 to be inactive")
	}
}

func TestRemoveLine_PassengerDisembarkation(t *testing.T) {
	sim := NewSimulator([]Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
	})

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	tr := &sim.State.Trains[0]

	// Train is past halfway mark on segment 0 (progress 0.8 -> nearest station is 1: Triangle)
	tr.Segment = 0
	tr.Progress = 0.8

	// Put passengers aboard:
	// Forced unloading due to line destruction must NOT count as a delivery reward.
	// Both passengers should alight to station 1 queue.
	tr.Passengers = []Passenger{
		{ID: 101, Destination: Triangle, SpawnTick: 1},
		{ID: 102, Destination: Square, SpawnTick: 1},
	}

	initialScore := sim.State.Score
	err := sim.ApplyAction(RemoveLine{LineID: 0})
	if err != nil {
		t.Fatalf("unexpected error removing line: %v", err)
	}

	// Forced unloading must NOT increment score
	if sim.State.Score != initialScore {
		t.Errorf("expected score %d (no fake delivery reward), got %d", initialScore, sim.State.Score)
	}

	// Both passengers queued at station 1
	st1 := &sim.State.Stations[1]
	if len(st1.Queue) != 2 {
		t.Errorf("expected 2 passengers in station 1 queue, got %d: %+v", len(st1.Queue), st1.Queue)
	}
}

func TestRemoveLine_LoopWrapAroundTunnelRefund(t *testing.T) {
	// Water between station 2 and station 0
	rivers := []RiverSegment{
		{From: Pos{X: 10, Y: 5}, To: Pos{X: -10, Y: 5}},
	}
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 5, Y: 10}, Alive: true, Capacity: 6},
	}
	sim := NewSimulatorWithRivers(stations, rivers, 42)

	initTunnels := sim.State.Resources.Tunnels
	sim.State.Resources.Grant(RewardTunnel) // tunnel for 1 -> 2
	sim.State.Resources.Grant(RewardTunnel) // tunnel for 2 -> 0 wrap-around

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})

	err := sim.ApplyAction(CloseLoop{LineID: 0, UseTunnel: true})
	if err != nil {
		t.Fatalf("unexpected error closing loop: %v", err)
	}

	if sim.State.Resources.Tunnels != initTunnels {
		t.Errorf("expected 0 net tunnels remaining before removal, got %d", sim.State.Resources.Tunnels)
	}

	// Remove loop line -> should refund both segment tunnel and loop wrap-around tunnel
	err = sim.ApplyAction(RemoveLine{LineID: 0})
	if err != nil {
		t.Fatalf("unexpected error removing loop line: %v", err)
	}

	if sim.State.Resources.Tunnels != initTunnels+2 {
		t.Errorf("expected refunded tunnels %d, got %d", initTunnels+2, sim.State.Resources.Tunnels)
	}
}

func TestShortenLine_FromFront(t *testing.T) {
	rivers := []RiverSegment{
		{From: Pos{X: 5, Y: -10}, To: Pos{X: 5, Y: 10}},
	}
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	}
	sim := NewSimulatorWithRivers(stations, rivers, 42)

	sim.State.Resources.Grant(RewardTunnel)
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}}) // uses tunnel
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})

	initTunnels := sim.State.Resources.Tunnels
	sim.State.Trains[0].Segment = 0
	sim.State.Trains[0].Progress = 0.5

	// Shorten from front -> removes station 0, refunds tunnel on segment 0->1
	err := sim.ApplyAction(ShortenLine{LineID: 0, FromFront: true})
	if err != nil {
		t.Fatalf("unexpected error shortening line from front: %v", err)
	}

	line := &sim.State.Lines[0]
	if len(line.Stations) != 2 || line.Stations[0] != 1 || line.Stations[1] != 2 {
		t.Errorf("expected stations [1, 2], got %+v", line.Stations)
	}

	if sim.State.Resources.Tunnels != initTunnels+1 {
		t.Errorf("expected tunnel refund, got %d", sim.State.Resources.Tunnels)
	}

	tr := &sim.State.Trains[0]
	if tr.Segment != 0 || tr.Direction != 1 {
		t.Errorf("expected train at segment 0 with direction 1, got seg=%d dir=%d", tr.Segment, tr.Direction)
	}
}

func TestShortenLine_FromBack(t *testing.T) {
	sim := NewSimulator([]Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	})

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})

	// Train is at tail station (segment 1)
	sim.State.Trains[0].Segment = 1
	sim.State.Trains[0].Progress = 0.9

	// Shorten from back -> removes station 2
	err := sim.ApplyAction(ShortenLine{LineID: 0, FromFront: false})
	if err != nil {
		t.Fatalf("unexpected error shortening line from back: %v", err)
	}

	line := &sim.State.Lines[0]
	if len(line.Stations) != 2 || line.Stations[0] != 0 || line.Stations[1] != 1 {
		t.Errorf("expected stations [0, 1], got %+v", line.Stations)
	}

	tr := &sim.State.Trains[0]
	if tr.Segment != 1 || tr.Direction != -1 {
		t.Errorf("expected train at new tail (seg=1) heading back (dir=-1), got seg=%d dir=%d", tr.Segment, tr.Direction)
	}
}

func TestShortenLine_Validation(t *testing.T) {
	sim := NewSimulator([]Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	})

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})

	// Cannot shorten line with <= 2 stations
	err := sim.ApplyAction(ShortenLine{LineID: 0, FromFront: true})
	if err == nil {
		t.Errorf("expected error shortening 2-station line, got nil")
	}

	// Extend to 3 stations then close loop
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})
	_ = sim.ApplyAction(CloseLoop{LineID: 0})

	// Cannot shorten loop line
	err = sim.ApplyAction(ShortenLine{LineID: 0, FromFront: true})
	if err == nil {
		t.Errorf("expected error shortening loop line, got nil")
	}
}

func TestActionMask_RemoveAndShorten(t *testing.T) {
	sim := NewSimulator([]Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 10, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 20, Y: 0}, Alive: true, Capacity: 6},
	})

	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1}})

	mask := sim.GetActionMask(nil)
	// Line 0 has 2 stations: RemoveLine should be true, ShortenLine should be false
	if !mask[RemoveLineOffset+0] {
		t.Errorf("expected RemoveLine for line 0 to be valid")
	}
	if mask[ShortenLineOffset+0*2+0] || mask[ShortenLineOffset+0*2+1] {
		t.Errorf("expected ShortenLine for 2-station line 0 to be false")
	}

	// Extend line 0 to 3 stations
	_ = sim.ApplyAction(ExtendLine{LineID: 0, StationID: 2})
	mask = sim.GetActionMask(nil)

	// Now ShortenLine should be true for both front and back
	if !mask[ShortenLineOffset+0*2+0] {
		t.Errorf("expected ShortenLine (front) to be valid for 3-station line")
	}
	if !mask[ShortenLineOffset+0*2+1] {
		t.Errorf("expected ShortenLine (back) to be valid for 3-station line")
	}

	// Close loop on line 0
	_ = sim.ApplyAction(CloseLoop{LineID: 0})
	mask = sim.GetActionMask(nil)

	// ShortenLine should be false for loop line
	if mask[ShortenLineOffset+0*2+0] || mask[ShortenLineOffset+0*2+1] {
		t.Errorf("expected ShortenLine for loop line to be false")
	}
	// RemoveLine remains valid
	if !mask[RemoveLineOffset+0] {
		t.Errorf("expected RemoveLine for loop line to be valid")
	}
}
