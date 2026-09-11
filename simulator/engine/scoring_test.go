package engine

import (
	"math"
	"testing"
)

func TestComputeStepRewardBreakdown_Identity(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	// Step reward breakdown with deliveredDelta = 3
	total, rb := sim.ComputeStepRewardBreakdown(3)

	if rb.Delivery != 3.0 {
		t.Errorf("expected Delivery == 3.0, got %f", rb.Delivery)
	}

	expectedSum := rb.Delivery + rb.Connectivity + rb.CrowdPenalty + rb.GameOver + rb.Redundancy + rb.LoopReversal + rb.TrackEfficiency
	if math.Abs(total-expectedSum) > 1e-9 {
		t.Errorf("total reward (%f) != sum of decomposed channels (%f)", total, expectedSum)
	}
	if math.Abs(rb.Total-total) > 1e-9 {
		t.Errorf("rb.Total (%f) != total (%f)", rb.Total, total)
	}
}

func TestScoringConfig_Ablations(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	// 1. Default config check
	cfg := sim.GetScoringConfig()
	if cfg.AlphaCrowdPenalty != 0.30 || cfg.BetaGameOverPenalty != 200.0 || cfg.ConnectivityBonus != 2.0 || cfg.TrackEfficiencyWeight != 0.01 || cfg.LinearCrowdPenalty {
		t.Errorf("unexpected default scoring config: %+v", cfg)
	}

	// 2. Ablation A: Disable Connectivity Bonus (0.0)
	sim.ScoringConfig = ScoringConfig{
		AlphaCrowdPenalty:   0.30,
		BetaGameOverPenalty: 200.0,
		ConnectivityBonus:   0.0,
		LinearCrowdPenalty:  false,
		Initialized:         true,
	}
	_, rbA := sim.ComputeStepRewardBreakdown(1)
	if rbA.Connectivity != 0.0 {
		t.Errorf("expected Connectivity == 0.0 for Ablation A, got %f", rbA.Connectivity)
	}

	// 3. Ablation B: Linear Crowd Penalty
	st := &Station{Capacity: 6, Queue: make([]Passenger, 12), Alive: true, OvercrowdingTimer: -1}
	quadPenalty := StationCrowdPenalty(st, false) // (6/6)^2 = 1.0
	linPenalty := StationCrowdPenalty(st, true)   // (6/6) = 1.0
	if math.Abs(quadPenalty-linPenalty) > 1e-9 {
		t.Errorf("expected equal for overflow 1.0, got quad=%f, lin=%f", quadPenalty, linPenalty)
	}
	// With 18 passengers: overflowRatio = 12/6 = 2.0. Quad = 4.0, Lin = 2.0
	st.Queue = make([]Passenger, 18)
	quadPenalty = StationCrowdPenalty(st, false)
	linPenalty = StationCrowdPenalty(st, true)
	if math.Abs(quadPenalty-4.0) > 1e-9 || math.Abs(linPenalty-2.0) > 1e-9 {
		t.Errorf("overflow 2.0: expected quad=4.0, lin=2.0; got quad=%f, lin=%f", quadPenalty, linPenalty)
	}

	// 4. Ablation C: BetaGameOverPenalty reduced 200 -> 50
	sim.ScoringConfig = ScoringConfig{
		AlphaCrowdPenalty:   0.30,
		BetaGameOverPenalty: 50.0,
		ConnectivityBonus:   2.0,
		LinearCrowdPenalty:  false,
		Initialized:         true,
	}
	sim.State.Alive = false
	_, rbC := sim.ComputeStepRewardBreakdown(0)
	if rbC.GameOver != -50.0 {
		t.Errorf("expected GameOver == -50.0 for Ablation C, got %f", rbC.GameOver)
	}
}

func TestStepMacroBreakdown(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	obs, reward, done, info, rb := sim.StepMacroBreakdown(nil, 1.0)
	if done {
		t.Errorf("expected done == false for step 0")
	}
	if info.StepTicks <= 0 {
		t.Errorf("expected StepTicks > 0, got %d", info.StepTicks)
	}
	if len(obs.StationKinds) == 0 {
		t.Errorf("expected stations in observation")
	}
	expectedTotal := rb.Delivery + rb.Connectivity + rb.CrowdPenalty + rb.GameOver + rb.Redundancy + rb.LoopReversal + rb.TrackEfficiency
	if math.Abs(reward-expectedTotal) > 1e-6 {
		t.Errorf("reward (%f) != breakdown sum (%f)", reward, expectedTotal)
	}
	if math.Abs(reward-rb.Total) > 1e-6 {
		t.Errorf("reward (%f) != rb.Total (%f)", reward, rb.Total)
	}
}

func TestTrackEfficiency_TotalTrackLength(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 30, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 30, Y: 40}, Alive: true, Capacity: 6},
	}
	sim := NewSimulatorWithWater(stations, nil, nil, 42)

	// Initially 0 track length
	if sim.TotalTrackLength() != 0.0 {
		t.Errorf("expected 0 track length initially, got %f", sim.TotalTrackLength())
	}

	// Line 0: [0, 1, 2] -> Seg 0-1 (dist 30) + Seg 1-2 (dist 40) = 70.0
	sim.State.Lines = []Line{
		{ID: 0, Stations: []int{0, 1, 2}, IsLoop: false},
	}
	if math.Abs(sim.TotalTrackLength()-70.0) > 1e-6 {
		t.Errorf("expected track length 70.0, got %f", sim.TotalTrackLength())
	}

	// Make Line 0 a loop: adds Seg 2-0 (dist 50) = 70 + 50 = 120.0
	sim.State.Lines[0].IsLoop = true
	if math.Abs(sim.TotalTrackLength()-120.0) > 1e-6 {
		t.Errorf("expected loop track length 120.0, got %f", sim.TotalTrackLength())
	}

	// Check TrackEfficiency penalty in ComputeStepRewardBreakdown:
	// R_track_efficiency = -0.01 * (120.0 / 100.0) = -0.012
	_, rb := sim.ComputeStepRewardBreakdown(0)
	expectedPenalty := -0.01 * (120.0 / 100.0)
	if math.Abs(rb.TrackEfficiency-expectedPenalty) > 1e-6 {
		t.Errorf("expected TrackEfficiency penalty %f, got %f", expectedPenalty, rb.TrackEfficiency)
	}
}

func TestStepMacroBreakdown_DurationScaling(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 0, Y: 0}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 30, Y: 0}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 30, Y: 40}, Alive: true, Capacity: 6},
	}
	sim := NewSimulatorWithWater(stations, nil, nil, 42)
	sim.State.Lines = []Line{
		{ID: 0, Stations: []int{0, 1, 2}, IsLoop: false},
	}
	sim.ScoringConfig = ScoringConfig{
		TrackEfficiencyWeight: 0.05,
		AlphaCrowdPenalty:     0.20,
		Initialized:           true,
	}

	// 1.0s step -> 30 ticks
	_, _, _, info1, rb1 := sim.StepMacroBreakdown(nil, 1.0)
	if math.Abs(info1.SimulationSeconds-1.0) > 1e-6 {
		t.Errorf("expected SimulationSeconds 1.0, got %f", info1.SimulationSeconds)
	}

	// 0.5s step -> 15 ticks
	_, _, _, info2, rb2 := sim.StepMacroBreakdown(nil, 0.5)
	if math.Abs(info2.SimulationSeconds-0.5) > 1e-6 {
		t.Errorf("expected SimulationSeconds 0.5, got %f", info2.SimulationSeconds)
	}

	// Track efficiency penalty should scale linearly with elapsed duration
	expectedRatio := 0.5 / 1.0
	actualRatio := rb2.TrackEfficiency / rb1.TrackEfficiency
	if math.Abs(actualRatio-expectedRatio) > 1e-4 {
		t.Errorf("expected track efficiency ratio %f, got %f (rb1=%f, rb2=%f)",
			expectedRatio, actualRatio, rb1.TrackEfficiency, rb2.TrackEfficiency)
	}
}


