package engine

const trainSpeed = 0.5

func (s *Simulator) moveTrains(dt float64) {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]

		if !tr.Active {
			continue
		}

		line := &s.State.Lines[tr.LineID]

		if line.Removed || len(line.Stations) < 2 {
			continue
		}

		tr.Progress += trainSpeed * dt

		// reached next station
		if tr.Progress >= 1.0 {
			tr.Progress = 0
			tr.Segment += tr.Direction

			last := len(line.Stations) - 1

			// bounce at ends
			if tr.Segment >= last {
				tr.Segment = last
				tr.Direction = -1
			}

			if tr.Segment <= 0 {
				tr.Segment = 0
				tr.Direction = 1
			}
		}
	}
}

func (s *Simulator) boardAndAlight() {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]

		if !tr.Active {
			continue
		}

		//care only if train is at station
		if tr.Progress != 0 {
			continue
		}

		line := &s.State.Lines[tr.LineID]
		stationID := line.Stations[tr.Segment]
		st := &s.State.Stations[stationID]

		// alight
		remaining := tr.Passengers[:0]

		for _, p := range tr.Passengers {
			if p == st.Kind {
				s.State.Score++
			} else {
				remaining = append(remaining, p)
			}
		}

		tr.Passengers = remaining

		// board
		for len(st.Queue) > 0 && len(tr.Passengers) < tr.Capacity {
			p := st.Queue[0]
			st.Queue = st.Queue[1:]
			tr.Passengers = append(tr.Passengers, p)
		}
	}
}
