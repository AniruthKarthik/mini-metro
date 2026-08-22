package engine

import "math"

const (
	AlphaCrowdPenalty    = 0.05
	BetaGameOverPenalty = 50.0
)

// StationCrowdPenalty calculates a non-linear overcrowding penalty for a station.
func StationCrowdPenalty(st *Station) float64 {
	qLen := len(st.Queue)
	if qLen <= st.Capacity {
		return 0.0
	}

	cap := float64(st.Capacity)
	if cap <= 0 {
		cap = 6.0
	}

	overflowRatio := float64(qLen-st.Capacity) / cap
	penalty := overflowRatio * overflowRatio

	if st.OvercrowdingTimer >= 0 {
		timerRatio := st.OvercrowdingTimer / overcrowdingGrace
		if timerRatio < 0 {
			timerRatio = 0
		}
		penalty += math.Exp(1.0 - timerRatio)
	}

	return penalty
}

// ComputeStepReward calculates instantaneous step reward R_t for RL.
func (s *Simulator) ComputeStepReward(deliveredDelta int) float64 {
	reward := float64(deliveredDelta)

	totalCrowdPenalty := 0.0
	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if st.Alive {
			totalCrowdPenalty += StationCrowdPenalty(st)
		}
	}

	reward -= AlphaCrowdPenalty * totalCrowdPenalty

	if !s.State.Alive {
		reward -= BetaGameOverPenalty
	}

	return reward
}

// updateScore is called each tick; passenger delivery score is tracked in boardAndAlight.
func (s *Simulator) updateScore() {}
