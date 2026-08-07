package engine

const maxQsize = 20

func (s *Simulator) checkGameOver() {
	for i := range s.State.Stations {
		st := &s.State.Stations[i]

		if !st.Alive {
			continue
		}

		if len(st.Queue) > maxQsize {
			s.State.Alive = false
			return
		}
	}
}
