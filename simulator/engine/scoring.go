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
	// PHASE-6: IsolatedStationPenalty — penalize each alive station with degree 0 (unserved).
	// Gives immediate feedback as soon as a new station spawns, preventing the agent from ignoring it.
	IsolatedStationPenalty = 0.10
	// PHASE-6: RedundantActionPenalty — penalize actions connecting stations already reachable.
	RedundantActionPenalty = 0.75
	// PHASE-6: ConnectIsolatedBonus — reward connecting an isolated station to the network.
	ConnectIsolatedBonus = 0.75
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

	// PHASE-6: Isolated station penalty
	// Penalize alive stations with degree 0 (unserved by any line).
	// Immediately creates a negative reward gradient as soon as a new station spawns,
	// teaching the policy not to leave stations unconnected.
	isolatedCount := 0
	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if st.Alive && s.stationDegree(i) == 0 {
			isolatedCount++
		}
	}
	reward -= IsolatedStationPenalty * float64(isolatedCount)

	// PHASE-4: connectivity bonus — reward reachable distinct-type station pairs.
	// Prevents the agent from adding redundant connections (extra lines between already-
	// connected stations yield no new reachable pairs, so no additional bonus).
	// Count unique (typeA < typeB) pairs where at least one station of typeA can reach
	// at least one station of typeB.
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
		reward += ConnectivityBonus * float64(reachablePairs) / 45.0
	}

	return reward
}

// updateScore is called each tick; passenger delivery score is tracked in boardAndAlight.
func (s *Simulator) updateScore() {}
