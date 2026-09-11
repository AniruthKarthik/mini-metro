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
