package engine

const trainSpeed = 0.5
const dwellTime = 2.0 // seconds a train pauses at each station for boarding/alighting

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
		for tr.Progress >= 1.0 {
			tr.Progress -= 1.0

			if line.IsLoop {
				// wrap around — no bounce, always one-way
				n := len(line.Stations)
				tr.Segment = (tr.Segment + tr.Direction + n) % n
			} else {
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

			tr.JustArrived = true
		}

		// pause at station; interchange stations board/alight faster
		if tr.JustArrived {
			stID := line.Stations[tr.Segment]
			if s.State.Stations[stID].IsInterchange {
				tr.DwellRemaining = dwellTime / 2
			} else {
				tr.DwellRemaining = dwellTime
			}
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

		// Board — compute A* routes once per destKind per station visit, then filter by NextLineID.
		// Capacity reservation is implicit: passengers only board the optimal line, not any reachable train.
		totalCapacity := tr.Capacity
		if tr.Carriages > 1 {
			totalCapacity += (tr.Carriages - 1) * 6
		}

		routeCache := make(map[StationKind]RouteInfo)
		getRoute := func(dest StationKind) RouteInfo {
			if r, ok := routeCache[dest]; ok {
				return r
			}
			r := FindOptimalRoute(&s.State.Graph, &s.State, stationID, dest)
			routeCache[dest] = r
			return r
		}

		remaining := st.Queue[:0]
		for _, p := range st.Queue {
			route := getRoute(p.Destination)
			canBoard := len(tr.Passengers) < totalCapacity &&
				(st.IsInterchange || (route.Reachable && route.NextLineID == tr.LineID))
			if canBoard {
				tr.Passengers = append(tr.Passengers, p)
			} else {
				remaining = append(remaining, p)
			}
		}
		st.Queue = remaining
		if len(st.Queue) == 0 {
			st.Queue = nil
		}
	}
}
