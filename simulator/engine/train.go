package engine

const trainSpeed = 0.5

type Train struct {
	ID             int
	LineID         int
	Segment        int // line between stations, explains position of the train in the series of lines
	Progress       float64
	Direction      int
	Capacity       int
	Carriages      int
	Passengers     []Passenger
	Active         bool
	JustArrived    bool
	DwellRemaining float64
}

func (s *Simulator) moveTrains(dt float64) {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]

		if !tr.Active {
			continue
		}

		if tr.DwellRemaining > 0 {
			tr.DwellRemaining -= dt
			if tr.DwellRemaining > 0 {
				continue
			}
			tr.DwellRemaining = 0
		}

		if tr.LineID < 0 || tr.LineID >= len(s.State.Lines) {
			continue
		}

		line := &s.State.Lines[tr.LineID]

		if line.Removed || len(line.Stations) < 2 {
			continue
		}

		tr.Progress += trainSpeed * dt

		// reached next station
		if tr.Progress >= 1.0 {
			tr.Progress -= 1.0
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

			tr.JustArrived = true
		}
	}
}

func (s *Simulator) boardAndAlight() {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]

		if !tr.Active || !tr.JustArrived {
			continue
		}

		tr.JustArrived = false

		if tr.LineID < 0 || tr.LineID >= len(s.State.Lines) {
			continue
		}

		line := &s.State.Lines[tr.LineID]
		if line.Removed || len(line.Stations) < 2 {
			continue
		}

		if tr.Segment < 0 || tr.Segment >= len(line.Stations) {
			continue
		}

		stationID := line.Stations[tr.Segment]
		if stationID < 0 || stationID >= len(s.State.Stations) {
			continue
		}

		st := &s.State.Stations[stationID]
		if !st.Alive {
			continue
		}

		// Alight
		remainingPassengers := tr.Passengers[:0]
		for _, p := range tr.Passengers {
			if p.Destination == st.Kind {
				s.State.Score++
			} else {
				remainingPassengers = append(remainingPassengers, p)
			}
		}
		tr.Passengers = remainingPassengers

		// Board
		totalCapacity := tr.Capacity
		if tr.Carriages > 1 {
			totalCapacity += (tr.Carriages - 1) * 6
		}

		for len(st.Queue) > 0 && len(tr.Passengers) < totalCapacity {
			p := st.Queue[0]
			st.Queue = st.Queue[1:]
			tr.Passengers = append(tr.Passengers, p)
		}
	}
}
