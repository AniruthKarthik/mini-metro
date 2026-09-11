package engine

import "math"

const (
	// PHASE-4: AlphaCrowdPenalty raised 0.05→0.30 (was 22× too weak vs delivery reward)
	AlphaCrowdPenalty = 0.30
	// PHASE-4: BetaGameOverPenalty raised 50→200 (game-over signal overwhelmed by cumulative rewards)
	BetaGameOverPenalty = 200.0
	// PHASE-4: ConnectivityBonus — reward per distinct (type-a, type-b) pair that can reach each other.
	// Gives positive gradient for building a connected network, not just spamming connections.
	// Scaled so a 3-station, 2-type network (~1 pair) gives +2.0/step vs ~1.0/step from delivery.
	ConnectivityBonus = 2.0
)

// ScoringConfig allows runtime configuration and ablation of reward components (P2-1, P2-2).
type ScoringConfig struct {
	AlphaCrowdPenalty     float64 // default: 0.30
	BetaGameOverPenalty   float64 // default: 200.0
	ConnectivityBonus     float64 // default: 2.0
	TrackEfficiencyWeight float64 // default: 0.01 (P2-2 continuous track sprawl penalty)
	LinearCrowdPenalty    bool    // default: false (quadratic overflow)
	Initialized           bool
}

func DefaultScoringConfig() ScoringConfig {
	return ScoringConfig{
		AlphaCrowdPenalty:     AlphaCrowdPenalty,
		BetaGameOverPenalty:   BetaGameOverPenalty,
		ConnectivityBonus:     ConnectivityBonus,
		TrackEfficiencyWeight: 0.01,
		LinearCrowdPenalty:    false,
		Initialized:           true,
	}
}

// GetScoringConfig returns s.ScoringConfig if initialized, or DefaultScoringConfig() otherwise.
func (s *Simulator) GetScoringConfig() ScoringConfig {
	if !s.ScoringConfig.Initialized {
		return DefaultScoringConfig()
	}
	return s.ScoringConfig
}

// RewardBreakdown holds the instantaneous decomposition of the step reward (P2-1, P2-2).
type RewardBreakdown struct {
	Delivery        float64
	Connectivity    float64
	CrowdPenalty    float64
	GameOver        float64
	Redundancy      float64
	LoopReversal    float64
	TrackEfficiency float64
	Disruption      float64
	Total           float64
}


// StationCrowdPenalty calculates an overcrowding penalty for a station.
func StationCrowdPenalty(st *Station, linear ...bool) float64 {
	qLen := len(st.Queue)
	if qLen <= st.Capacity {
		return 0.0
	}

	cap := float64(st.Capacity)
	if cap <= 0 {
		cap = 6.0
	}

	overflowRatio := float64(qLen-st.Capacity) / cap
	var penalty float64
	if len(linear) > 0 && linear[0] {
		penalty = overflowRatio
	} else {
		penalty = overflowRatio * overflowRatio
	}

	if st.OvercrowdingTimer >= 0 {
		penalty += math.Exp(OvercrowdingProgress(st))
	}

	return penalty
}

// CanReach returns true if stationID u can reach stationID v via the current rail network.
// Uses a simple BFS on the adjacency graph. O(N+E), called once per step so performance is fine.
func (s *Simulator) CanReach(u, v int) bool {
	if u == v {
		return true
	}
	N := len(s.State.Stations)
	if u < 0 || v < 0 || u >= N || v >= N {
		return false
	}
	visited := make([]bool, N)
	queue := []int{u}
	visited[u] = true
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range s.State.Graph.Neighbours(cur) {
			if nb == v {
				return true
			}
			if !visited[nb] {
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	return false
}

// ComputeStepRewardBreakdown computes both the total reward and its decomposed constituents (P2-1).
func (s *Simulator) ComputeStepRewardBreakdown(deliveredDelta int) (float64, RewardBreakdown) {
	cfg := s.GetScoringConfig()

	rb := RewardBreakdown{}

	// 1. Delivery reward
	rb.Delivery = float64(deliveredDelta)

	// 2. Crowd penalty
	totalCrowdPenalty := 0.0
	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if st.Alive {
			totalCrowdPenalty += StationCrowdPenalty(st, cfg.LinearCrowdPenalty)
		}
	}
	rb.CrowdPenalty = -cfg.AlphaCrowdPenalty * totalCrowdPenalty

	// 3. Game over penalty
	if !s.State.Alive {
		rb.GameOver = -cfg.BetaGameOverPenalty
	}

	// 4. Connectivity bonus & redundancy penalty
	s.rebuildGraphIfNeeded()
	N := len(s.State.Stations)
	if N > 1 {
		// Build per-type representative station IDs
		typeRep := [10]int{}
		for k := range typeRep {
			typeRep[k] = -1
		}
		for i := 0; i < N; i++ {
			st := &s.State.Stations[i]
			if st.Alive && typeRep[int(st.Kind)] < 0 {
				typeRep[int(st.Kind)] = i
			}
		}
		reachablePairs := 0
		for a := 0; a < 10; a++ {
			if typeRep[a] < 0 {
				continue
			}
			for b := a + 1; b < 10; b++ {
				if typeRep[b] < 0 {
					continue
				}
				if s.CanReach(typeRep[a], typeRep[b]) {
					reachablePairs++
				}
			}
		}
		// Normalize by max possible pairs (45) so bonus stays in [0, ~2.0]
		rb.Connectivity = cfg.ConnectivityBonus * float64(reachablePairs) / 45.0

		// P1-2: Redundant network expansion penalty.
		// Penalize stations served by >2 lines unless upgraded to an interchange hub.
		linesServingStation := make([]int, N)
		for _, line := range s.State.Lines {
			if line.Removed {
				continue
			}
			seenInLine := make(map[int]bool)
			for _, stID := range line.Stations {
				if stID >= 0 && stID < N && !seenInLine[stID] {
					linesServingStation[stID]++
					seenInLine[stID] = true
				}
			}
		}
		redundancyPenalty := 0.0
		for i := 0; i < N; i++ {
			st := &s.State.Stations[i]
			if st.Alive && !st.IsInterchange && linesServingStation[i] > 2 {
				redundancyPenalty += 0.05 * float64(linesServingStation[i]-2)
			}
		}
		rb.Redundancy = -redundancyPenalty
	}

	// 5. Loop rapid reversal penalty
	if s.loopTogglePenalty > 0 {
		rb.LoopReversal = -s.loopTogglePenalty
		s.loopTogglePenalty = 0.0
	}

	// 6. Continuous track-mileage regularization penalty (P2-2)
	// R_track_efficiency = -w_track * sum_{e in Network} (Distance(e) / 100.0)
	if cfg.TrackEfficiencyWeight > 0 {
		rb.TrackEfficiency = -cfg.TrackEfficiencyWeight * (s.TotalTrackLength() / 100.0)
	}

	// 7. Operational disruption penalty (structural line deletion / severe disruption)
	if s.disruptionPenalty > 0 {
		rb.Disruption = -s.disruptionPenalty
		s.disruptionPenalty = 0.0
	}

	rb.Total = rb.Delivery + rb.Connectivity + rb.CrowdPenalty + rb.GameOver + rb.Redundancy + rb.LoopReversal + rb.TrackEfficiency + rb.Disruption
	return rb.Total, rb

}

// TotalTrackLength calculates the sum of all physical track segment lengths across all active lines (P2-2).
func (s *Simulator) TotalTrackLength() float64 {
	total := 0.0
	N := len(s.State.Stations)
	for _, line := range s.State.Lines {
		if line.Removed || len(line.Stations) < 2 {
			continue
		}
		for i := 0; i+1 < len(line.Stations); i++ {
			u := line.Stations[i]
			v := line.Stations[i+1]
			if u >= 0 && u < N && v >= 0 && v < N {
				total += distance(s.State.Stations[u].Pos, s.State.Stations[v].Pos)
			}
		}
		if line.IsLoop && len(line.Stations) >= 2 {
			u := line.Stations[len(line.Stations)-1]
			v := line.Stations[0]
			if u >= 0 && u < N && v >= 0 && v < N {
				total += distance(s.State.Stations[u].Pos, s.State.Stations[v].Pos)
			}
		}
	}
	return total
}

// ComputeStepReward calculates instantaneous step reward R_t for RL.
func (s *Simulator) ComputeStepReward(deliveredDelta int) float64 {
	total, _ := s.ComputeStepRewardBreakdown(deliveredDelta)
	return total
}

// updateScore is called each tick; passenger delivery score is tracked in boardAndAlight.
func (s *Simulator) updateScore() {}
