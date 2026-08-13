package engine

import (
	"math/rand"
)

const (
	baseSpawnRate     = 0.03   // 1 passenger every ~33 seconds per station
	spawnAccelPerTick = 0.000005 // gradual acceleration over game ticks
	maxSpawnRate      = 0.3    // max spawn rate cap
)

// CurrentSpawnRate returns the passenger spawn rate (passengers/sec per station),
// which accelerates gently over simulation time (s.State.Tick).
func (s *Simulator) CurrentSpawnRate() float64 {
	rate := baseSpawnRate + float64(s.State.Tick)*spawnAccelPerTick
	if rate > maxSpawnRate {
		return maxSpawnRate
	}
	return rate
}

// destinationWeights defines passenger attraction demand for each station kind.
var destinationWeights = map[StationKind]int{
	Circle:   2,
	Triangle: 3,
	Square:   4,
	Star:     8,
	Pentagon: 8,
	Gem:      8,
	Sector:   8,
	Cross:    8,
	Drop:     8,
	Oval:     8,
}

// sampleDestinationKind selects a destination StationKind for a passenger spawning at originKind,
// prioritizing active station kinds present on the map weighted by destinationWeights.
func sampleDestinationKind(state *GameState, originKind StationKind) StationKind {
	activeKinds := make(map[StationKind]bool)
	for i := range state.Stations {
		st := &state.Stations[i]
		if st.Alive && st.Kind != originKind {
			activeKinds[st.Kind] = true
		}
	}

	totalWeight := 0
	candidates := make([]StationKind, 0, len(activeKinds))
	for k := range activeKinds {
		w := destinationWeights[k]
		if w <= 0 {
			w = 1
		}
		totalWeight += w
		candidates = append(candidates, k)
	}

	if len(candidates) == 0 {
		for k, w := range destinationWeights {
			if k != originKind {
				totalWeight += w
				candidates = append(candidates, k)
			}
		}
	}

	if totalWeight <= 0 || len(candidates) == 0 {
		return Circle
	}

	r := rand.Intn(totalWeight)
	for _, k := range candidates {
		w := destinationWeights[k]
		if w <= 0 {
			w = 1
		}
		r -= w
		if r < 0 {
			return k
		}
	}
	return candidates[0]
}

func (s *Simulator) spawnPassengers(dt float64) {
	rate := s.CurrentSpawnRate()
	prob := rate * dt
	if prob > 1.0 {
		prob = 1.0
	}

	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if !st.Alive {
			continue
		}
		if rand.Float64() < prob {
			dest := sampleDestinationKind(&s.State, st.Kind)
			st.Queue = append(st.Queue, Passenger{
				Origin:      st.ID,
				Destination: dest,
				SpawnTick:   s.State.Tick,
			})
		}
	}
}
