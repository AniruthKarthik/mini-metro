package engine

import (
	"testing"
)

func TestRewardObservationSerialization(t *testing.T) {
	if GlobalFeatureDim != 23 {
		t.Fatalf("expected GlobalFeatureDim to be 23, got %d", GlobalFeatureDim)
	}

	sim := NewSimulatorWithMap(LondonMap(), 42)

	// 1. Initially no reward pending: indices 13..22 must be 0.0
	obs := sim.VectorizedObservation()
	if obs.GlobalDim != 23 {
		t.Fatalf("expected obs.GlobalDim == 23, got %d", obs.GlobalDim)
	}
	if len(obs.Globals) != 23 {
		t.Fatalf("expected len(obs.Globals) == 23, got %d", len(obs.Globals))
	}
	if obs.Globals[11] != 0.0 {
		t.Fatalf("expected pending reward flag globals[11] == 0.0, got %f", obs.Globals[11])
	}
	for i := 13; i < 23; i++ {
		if obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0 when no reward pending, got %f", i, obs.Globals[i])
		}
	}

	// 2. Set Card 0 = Line (0), Card 1 = Tunnel (2)
	sim.State.PendingRewardChoices = []RewardType{RewardLine, RewardTunnel}
	obs = sim.VectorizedObservation()
	if obs.Globals[11] != 1.0 {
		t.Fatalf("expected globals[11] == 1.0, got %f", obs.Globals[11])
	}
	if obs.Globals[13] != 1.0 {
		t.Fatalf("expected globals[13] == 1.0 for Card 0 Line, got %f", obs.Globals[13])
	}
	for i := 14; i < 18; i++ {
		if obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0, got %f", i, obs.Globals[i])
		}
	}
	if obs.Globals[20] != 1.0 {
		t.Fatalf("expected globals[20] == 1.0 for Card 1 Tunnel, got %f", obs.Globals[20])
	}
	for i := 18; i < 23; i++ {
		if i != 20 && obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0, got %f", i, obs.Globals[i])
		}
	}

	// 3. Permutation test: Swap to Card 0 = Tunnel (2), Card 1 = Line (0)
	sim.State.PendingRewardChoices = []RewardType{RewardTunnel, RewardLine}
	obs = sim.VectorizedObservation()
	if obs.Globals[15] != 1.0 {
		t.Fatalf("expected globals[15] == 1.0 for swapped Card 0 Tunnel, got %f", obs.Globals[15])
	}
	for i := 13; i < 18; i++ {
		if i != 15 && obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0, got %f", i, obs.Globals[i])
		}
	}
	if obs.Globals[18] != 1.0 {
		t.Fatalf("expected globals[18] == 1.0 for swapped Card 1 Line, got %f", obs.Globals[18])
	}
	for i := 19; i < 23; i++ {
		if obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0, got %f", i, obs.Globals[i])
		}
	}

	// 4. Test Carriage=3 and Interchange=4
	sim.State.PendingRewardChoices = []RewardType{RewardCarriage, RewardInterchange}
	obs = sim.VectorizedObservation()
	if obs.Globals[16] != 1.0 {
		t.Fatalf("expected globals[16] == 1.0 for Card 0 Carriage, got %f", obs.Globals[16])
	}
	if obs.Globals[22] != 1.0 {
		t.Fatalf("expected globals[22] == 1.0 for Card 1 Interchange, got %f", obs.Globals[22])
	}

	// 5. Test clearing
	sim.State.PendingRewardChoices = nil
	obs = sim.VectorizedObservation()
	if obs.Globals[11] != 0.0 {
		t.Fatalf("expected globals[11] == 0.0 after clearing, got %f", obs.Globals[11])
	}
	for i := 13; i < 23; i++ {
		if obs.Globals[i] != 0.0 {
			t.Fatalf("expected globals[%d] == 0.0 after clearing, got %f", i, obs.Globals[i])
		}
	}
}

func TestRedundancyPenalty(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)
	sim.ScoringConfig.TrackEfficiencyWeight = 0.0

	// Ensure at least 4 lines exist in State
	for len(sim.State.Lines) < 4 {
		sim.State.Lines = append(sim.State.Lines, Line{ID: len(sim.State.Lines)})
	}

	// 1. Station 0 served by 2 lines (no penalty)
	sim.State.Lines[0].Stations = []int{0, 1}
	sim.State.Lines[1].Stations = []int{0, 2}
	sim.State.Lines[2].Stations = []int{1, 2}
	sim.State.Stations[0].IsInterchange = false

	r2 := sim.ComputeStepReward(0)

	// 2. Station 0 served by 3 lines (regular station: penalty = -0.05 * 1 = -0.05)
	sim.State.Lines[2].Stations = []int{0, 2}
	r3 := sim.ComputeStepReward(0)

	penalty3 := r2 - r3
	if penalty3 < 0.049 || penalty3 > 0.051 {
		t.Fatalf("expected redundancy penalty ~0.05 for 3 lines, got diff %f", penalty3)
	}

	// 3. Station 0 upgraded to interchange (exempt from redundancy penalty)
	sim.State.Stations[0].IsInterchange = true
	r3Hub := sim.ComputeStepReward(0)

	if r3Hub != r2 {
		t.Fatalf("expected interchange hub to incur zero redundancy penalty, got %f vs %f", r3Hub, r2)
	}

	// 4. Station 0 served by 4 lines without interchange (penalty = -0.05 * 2 = -0.10)
	sim.State.Stations[0].IsInterchange = false
	sim.State.Lines[3].Stations = []int{0, 3}
	r4 := sim.ComputeStepReward(0)

	penalty4 := r2 - r4
	if penalty4 < 0.099 || penalty4 > 0.101 {
		t.Fatalf("expected redundancy penalty ~0.10 for 4 lines, got diff %f", penalty4)
	}
}

func TestLoopHysteresisAndCooldown(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	// Ensure line 0 has 3 stations (0, 1, 2)
	sim.State.Lines = []Line{
		{ID: 0, Stations: []int{0, 1, 2}},
	}
	sim.State.Tick = 100

	mask := make([]bool, MaxActionSpaceSize())
	sim.GetActionMask(mask)

	closeAction := CloseLoopOffset + 0
	openAction := OpenLoopOffset + 0

	// 1. Initially CloseLoop is valid, OpenLoop is invalid
	if !mask[closeAction] {
		t.Fatalf("expected CloseLoop to be valid for 3-station line initially")
	}
	if mask[openAction] {
		t.Fatalf("expected OpenLoop to be invalid for open line")
	}

	// 2. Apply CloseLoop
	err := sim.ApplyAction(CloseLoop{LineID: 0})
	if err != nil {
		t.Fatalf("CloseLoop failed: %v", err)
	}
	if !sim.State.Lines[0].IsLoop {
		t.Fatalf("expected line to be loop")
	}

	// 3. Immediately after CloseLoop (tick 100), BOTH CloseLoop and OpenLoop must be MASKED OUT by cooldown!
	sim.GetActionMask(mask)
	if mask[closeAction] {
		t.Fatalf("expected CloseLoop to be masked during cooldown")
	}
	if mask[openAction] {
		t.Fatalf("expected OpenLoop to be masked during cooldown (tick 100)")
	}

	// 4. Advance 899 ticks (tick 999 < 100 + 900) -> still masked
	sim.State.Tick = 999
	sim.GetActionMask(mask)
	if mask[openAction] {
		t.Fatalf("expected OpenLoop to remain masked at tick 999")
	}

	// 5. Advance to tick 1000 (100 + 900) -> cooldown expires! OpenLoop becomes valid
	sim.State.Tick = 1000
	sim.GetActionMask(mask)
	if !mask[openAction] {
		t.Fatalf("expected OpenLoop to become valid at tick 1000")
	}

	// 6. Toggling back at tick 1000 (< 100 + 1800 ticks / 60s) must incur the rapid reversal penalty (-0.50)
	rBefore := sim.ComputeStepReward(0)
	err = sim.ApplyAction(OpenLoop{LineID: 0})
	if err != nil {
		t.Fatalf("OpenLoop failed: %v", err)
	}
	rAfter := sim.ComputeStepReward(0)
	diff := rBefore - rAfter
	if diff < 0.49 || diff > 0.51 {
		t.Fatalf("expected rapid reversal penalty ~0.50, got diff %f", diff)
	}
}


