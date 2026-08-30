package engine

import (
	"math/rand"
)

const (
	baseSpawnRate       = 0.04    // 1 passenger every ~25 seconds per station
	spawnAccelPerSecond = 0.00045 // equivalent to the old 30 TPS tick ramp
	maxSpawnRate        = 0.4     // max spawn rate cap
)

// CurrentSpawnRate returns the passenger spawn rate (passengers/sec per station),
// which accelerates gently over elapsed game time.
func (s *Simulator) CurrentSpawnRate() float64 {
	rate := baseSpawnRate + s.State.GameTimeSeconds*spawnAccelPerSecond
	if rate > maxSpawnRate {
		return maxSpawnRate
	}
	return rate
}

// destinationWeights defines relative passenger attraction for each station kind.
// All shapes use equal weight: bottlenecks arise from station scarcity, not
// inflated intrinsic demand (as per official Mini Metro mechanics).
var destinationWeights = map[StationKind]int{
	Circle:   1,
	Triangle: 1,
	Square:   1,
	Star:     1,
	Pentagon: 1,
	Gem:      1,
	Sector:   1,
	Cross:    1,
	Drop:     1,
	Oval:     1,
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
		return originKind
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
			if dest == st.Kind {
				continue
			}
			id := s.State.NextPassengerID
			s.State.NextPassengerID++
			st.Queue = append(st.Queue, Passenger{
				ID:          id,
				Destination: dest,
				SpawnTick:   s.State.Tick,
			})
		}
	}
}
