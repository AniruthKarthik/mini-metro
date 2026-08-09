package engine

const overcrowdingGrace = 100.0 // ticks before an overcrowded station triggers game over

// checkGameOver ticks overcrowding timers and ends the game if any expire.
func (s *Simulator) checkGameOver() {
	if !s.State.Alive {
		return
	}

	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if !st.Alive {
			continue
		}

		if len(st.Queue) > st.Capacity {
			if st.OvercrowdingTimer < 0 {
				st.OvercrowdingTimer = overcrowdingGrace
			} else {
				st.OvercrowdingTimer--
				if st.OvercrowdingTimer <= 0 {
					s.State.Alive = false
					return
				}
			}
		} else {
			st.OvercrowdingTimer = -1
		}
	}
}
