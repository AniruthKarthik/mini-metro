package engine

import "math/rand"

func spawnRate() float64 {
	return 0.2
}

const numStationKinds = 5

func randomOtherKind(kind StationKind) StationKind {
	for {
		k := StationKind(rand.Intn(numStationKinds))
		if k != kind {
			return k
		}
	}
}

func (s *Simulator) spawnPassengers(dt float64) {
	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if !st.Alive {
			continue
		}
		if rand.Float64() < spawnRate()*dt {
			dest := randomOtherKind(st.Kind)
			st.Queue = append(st.Queue, dest)
		}
	}
}
