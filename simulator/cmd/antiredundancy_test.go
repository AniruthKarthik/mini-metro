package main

import (
	"testing"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func TestAntiRedundancyDuplicateLineMasking(t *testing.T) {
	// Setup map with 3 starter stations
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 10, Y: 10}, Alive: true, Capacity: 6},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 20, Y: 10}, Alive: true, Capacity: 6},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 30, Y: 10}, Alive: true, Capacity: 6},
	})

	maskSize := engine.MaxActionSpaceSize()
	mask := make([]bool, maskSize)

	// Step 1: In the beginning, stations 0, 1, 2 are isolated (degree 0)
	sim.GetActionMask(mask)

	// Find action ID for AddLine(0, 1)
	addLine01, ok := engine.ActionToIndex(engine.AddLine{Stations: []int{0, 1}})
	if !ok || !mask[addLine01] {
		t.Fatalf("expected AddLine(0, 1) to be valid initially")
	}

	// Apply AddLine(0, 1) -> creates Line 0
	err := sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("failed to add line 0: %v", err)
	}

	// Step 2: Now direct segment (0, 1) exists on Line 0.
	// AddLine(0, 1) MUST be masked out (false)!
	sim.GetActionMask(mask)
	if mask[addLine01] {
		t.Errorf("expected AddLine(0, 1) to be MASKED FALSE after line 0 connects (0, 1)")
	}

	// Station 2 is still isolated (degree 0).
	// AddLine(0, 2) or AddLine(1, 2) should be allowed since station 2 is isolated.
	addLine02, _ := engine.ActionToIndex(engine.AddLine{Stations: []int{0, 2}})
	if !mask[addLine02] {
		t.Errorf("expected AddLine(0, 2) to be valid when station 2 is isolated")
	}

	// Apply AddLine(0, 2) -> creates Line 1
	err = sim.ApplyAction(engine.AddLine{Stations: []int{0, 2}})
	if err != nil {
		t.Fatalf("failed to add line 1: %v", err)
	}

	// Step 3: Now all 3 stations (0, 1, 2) are connected in 1 component via lines 0 and 1.
	// Segments are: (0, 1) and (0, 2).
	// None of the 3 stations are isolated.
	sim.GetActionMask(mask)

	// AddLine(0, 1) must be masked (direct track exists)
	if mask[addLine01] {
		t.Errorf("AddLine(0, 1) should be masked")
	}
	// AddLine(0, 2) must be masked (direct track exists)
	if mask[addLine02] {
		t.Errorf("AddLine(0, 2) should be masked")
	}
	// AddLine(1, 2) must ALSO be masked because all stations are already reachable (CanReach=true)
	// and no isolated station exists!
	addLine12, _ := engine.ActionToIndex(engine.AddLine{Stations: []int{1, 2}})
	if mask[addLine12] {
		t.Errorf("AddLine(1, 2) should be MASKED when all stations are already reachable and not isolated")
	}
}

func TestLoopActionCooldown(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 10, Y: 10}, Alive: true, Capacity: 6},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 20, Y: 10}, Alive: true, Capacity: 6},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 20}, Alive: true, Capacity: 6},
	})

	// Add a 3-station line
	_ = sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	_ = sim.ApplyAction(engine.ExtendLine{LineID: 0, StationID: 2, FromFront: false})

	maskSize := engine.MaxActionSpaceSize()
	mask := make([]bool, maskSize)

	sim.GetActionMask(mask)
	closeLoop0, _ := engine.ActionToIndex(engine.CloseLoop{LineID: 0})
	openLoop0, _ := engine.ActionToIndex(engine.OpenLoop{LineID: 0})

	if !mask[closeLoop0] {
		t.Fatalf("expected CloseLoop to be available for 3-station line")
	}

	// Close the loop
	err := sim.ApplyAction(engine.CloseLoop{LineID: 0})
	if err != nil {
		t.Fatalf("failed to close loop: %v", err)
	}

	// Immediately after closing loop, cooldown must mask OpenLoop
	sim.GetActionMask(mask)
	if mask[openLoop0] {
		t.Errorf("expected OpenLoop to be MASKED immediately after CloseLoop due to cooldown")
	}

	// Advance time past cooldown (10 seconds)
	sim.State.GameTimeSeconds += 11.0
	sim.GetActionMask(mask)
	if !mask[openLoop0] {
		t.Errorf("expected OpenLoop to be unmasked after 10s cooldown expires")
	}
}

func TestIsolatedStationPenaltyAndConnectionBonus(t *testing.T) {
	sim := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 10, Y: 10}, Alive: true, Capacity: 6},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 20, Y: 10}, Alive: true, Capacity: 6},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 30, Y: 10}, Alive: true, Capacity: 6},
	})

	// All 3 stations isolated -> isolated penalty should be 3 * 0.10 = 0.30
	r0 := sim.ComputeStepReward(0)

	// Now connect stations 0 and 1 via StepMacro
	_, rStep, _, _ := sim.StepMacro(engine.AddLine{Stations: []int{0, 1}}, 1.0)

	// rStep should reflect the ConnectIsolatedBonus (+0.75)
	if rStep <= r0 {
		t.Errorf("expected step reward with ConnectIsolatedBonus to exceed baseline, got rStep=%f, r0=%f", rStep, r0)
	}
}
