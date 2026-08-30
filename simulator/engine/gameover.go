package engine

const (
	overcrowdingCountdownSeconds = 45.0
	overcrowdingGraceSeconds     = 2.0
	overcrowdingGrace            = overcrowdingCountdownSeconds
	overcrowdingFailureSeconds   = overcrowdingCountdownSeconds + overcrowdingGraceSeconds
)

func (s *Simulator) trainAtStation(stationID int) bool {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if !tr.Active || tr.LineID < 0 || tr.LineID >= len(s.State.Lines) {
			continue
		}
		line := &s.State.Lines[tr.LineID]
		if line.Removed || tr.Segment < 0 || tr.Segment >= len(line.Stations) {
			continue
		}
		if line.Stations[tr.Segment] == stationID && tr.DwellRemaining > 0 {
			return true
		}
	}
	return false
}

func OvercrowdingProgress(st *Station) float64 {
	if st.OvercrowdingTimer < 0 {
		return 0
	}
	progress := 1.0 - ((st.OvercrowdingTimer - overcrowdingGraceSeconds) / overcrowdingCountdownSeconds)
	if progress < 0 {
		return 0
	}
	if progress > 1 {
		return 1
	}
	return progress
}

// checkGameOver advances overcrowding timers and ends the game if any expire.
func (s *Simulator) checkGameOver(dt float64) {
	if !s.State.Alive {
		return
	}
	if dt <= 0 {
		return
	}

	for i := range s.State.Stations {
		st := &s.State.Stations[i]
		if !st.Alive {
			continue
		}

		if len(st.Queue) > st.Capacity {
			if st.OvercrowdingTimer < 0 {
				st.OvercrowdingTimer = overcrowdingFailureSeconds
			} else if !s.trainAtStation(st.ID) {
				st.OvercrowdingTimer -= dt
			}
			if st.OvercrowdingTimer <= 0 {
				s.State.Alive = false
				return
			}
		} else {
			if st.OvercrowdingTimer >= 0 {
				st.OvercrowdingTimer += dt
				if st.OvercrowdingTimer >= overcrowdingFailureSeconds {
					st.OvercrowdingTimer = -1
				}
			}
		}
	}
}
