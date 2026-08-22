package engine

import (
	"math/rand"
)

const (
	baseSpawnRate     = 0.04    // 1 passenger every ~25 seconds per station
	spawnAccelPerTick = 0.000015 // gentle acceleration over game ticks
	maxSpawnRate      = 0.4     // max spawn rate cap
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
func sampleDestinationKind(state *GameState, originKind StationKind, rng *rand.Rand) StationKind {
	var activeKinds [16]StationKind
	numActive := 0
	for i := range state.Stations {
		st := &state.Stations[i]
		if st.Alive && st.Kind != originKind {
			duplicate := false
			for j := 0; j < numActive; j++ {
				if activeKinds[j] == st.Kind {
					duplicate = true
					break
				}
			}
			if !duplicate && numActive < len(activeKinds) {
				activeKinds[numActive] = st.Kind
				numActive++
			}
		}
	}

	totalWeight := 0
	for j := 0; j < numActive; j++ {
		w := destinationWeights[activeKinds[j]]
		if w <= 0 {
			w = 1
		}
		totalWeight += w
	}

	if numActive == 0 {
		var fallback [16]StationKind
		numFallback := 0
		for k, w := range destinationWeights {
			if k != originKind {
				totalWeight += w
				if numFallback < len(fallback) {
					fallback[numFallback] = k
					numFallback++
				}
			}
		}
		if totalWeight <= 0 || numFallback == 0 {
			return Circle
		}
		r := rng.Intn(totalWeight)
		for j := 0; j < numFallback; j++ {
			w := destinationWeights[fallback[j]]
			if w <= 0 {
				w = 1
			}
			r -= w
			if r < 0 {
				return fallback[j]
			}
		}
		return fallback[0]
	}

	if totalWeight <= 0 {
		return Circle
	}

	r := rng.Intn(totalWeight)
	for j := 0; j < numActive; j++ {
		w := destinationWeights[activeKinds[j]]
		if w <= 0 {
			w = 1
		}
		r -= w
		if r < 0 {
			return activeKinds[j]
		}
	}
	return activeKinds[0]
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
		if s.RNG().Float64() < prob {
			dest := sampleDestinationKind(&s.State, st.Kind, s.RNG())
			st.Queue = append(st.Queue, Passenger{
				Origin:      st.ID,
				Destination: dest,
				SpawnTick:   s.State.Tick,
			})
		}
	}
}
