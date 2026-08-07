package engine

import "math/rand"

func spawnRate() float64 {
	return 0.2
}

func randomOtherKind(kind int) int {
	for {
		k := rand.Intn(3)
		if k != kind {
			return k
		}
	}
}

func (s *Simulator) spawnPassengers(dt float64) {
	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if rand.Float64() < spawnRate()*dt {
			dest := randomOtherKind(st.Kind)
			st.Queue = append(st.Queue, dest)
		}
	}
}
