package engine

import (
	"math/rand"
)

const (
	baseSpawnRate     = 0.2
	spawnAccelPerTick = 0.0001
	maxSpawnRate      = 2.0
)

// CurrentSpawnRate returns the passenger spawn rate (passengers/sec per station),
// which accelerates over simulation time (s.State.Tick).
func (s *Simulator) CurrentSpawnRate() float64 {
	rate := baseSpawnRate + float64(s.State.Tick)*spawnAccelPerTick
	if rate > maxSpawnRate {
		return maxSpawnRate
	}
	return rate
}

const numStationKinds = 10

func randomOtherKind(kind StationKind) StationKind {
	for {
		k := StationKind(rand.Intn(numStationKinds))
		if k != kind {
			return k
		}
	}
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
			dest := randomOtherKind(st.Kind)
			st.Queue = append(st.Queue, Passenger{
				Origin:      st.ID,
				Destination: dest,
				SpawnTick:   s.State.Tick,
			})
		}
	}
}

