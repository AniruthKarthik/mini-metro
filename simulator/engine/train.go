package engine

import "math"

const trainSpeed = 1.8
const dwellTime = 0.4 // seconds a train pauses at each station for boarding/alighting

type Train struct {
	ID             int
	LineID         int
	Segment        int // line between stations
	Progress       float64
	Direction      int
	Capacity       int
	Carriages      int
	Passengers     []Passenger
	Active         bool
	JustArrived    bool
	DwellRemaining float64
}

// velocityProfile returns a smooth acceleration/deceleration multiplier in [0.4, 1.0] based on segment progress p in [0, 1].
func velocityProfile(p float64) float64 {
	if p < 0.0 {
		p = 0.0
	}
	if p > 1.0 {
		p = 1.0
	}
	if p < 0.2 {
		return 0.4 + 0.6*math.Sin((p/0.2)*(math.Pi/2.0))
	}
	if p > 0.8 {
		return 0.4 + 0.6*math.Sin(((1.0-p)/0.2)*(math.Pi/2.0))
	}
	return 1.0
}

// trackCornerMultiplier calculates the turn angle slowdown factor in [0.6, 1.0] at stationIndex on line.
func trackCornerMultiplier(state *GameState, line *Line, stationIndex int, dir int) float64 {
	if line.Removed || len(line.Stations) < 3 {
		return 1.0
	}
	N := len(line.Stations)
	if stationIndex < 0 || stationIndex >= N {
		return 1.0
	}

	var prevSeg, nextSeg int
	if line.IsLoop {
		prevSeg = (stationIndex - dir + N) % N
		nextSeg = (stationIndex + dir + N) % N
	} else {
		prevSeg = stationIndex - dir
		nextSeg = stationIndex + dir
		if prevSeg < 0 || prevSeg >= N || nextSeg < 0 || nextSeg >= N {
			return 0.7
		}
	}

	if prevSeg < 0 || prevSeg >= len(line.Stations) || nextSeg < 0 || nextSeg >= len(line.Stations) {
		return 1.0
	}

	uSt := line.Stations[prevSeg]
	cSt := line.Stations[stationIndex]
	vSt := line.Stations[nextSeg]
	if uSt < 0 || uSt >= len(state.Stations) || cSt < 0 || cSt >= len(state.Stations) || vSt < 0 || vSt >= len(state.Stations) {
		return 1.0
	}

	pPrev := state.Stations[uSt].Pos
	pCur := state.Stations[cSt].Pos
	pNext := state.Stations[vSt].Pos

	ux, uy := pCur.X-pPrev.X, pCur.Y-pPrev.Y
	vx, vy := pNext.X-pCur.X, pNext.Y-pCur.Y

	uLen := math.Sqrt(ux*ux + uy*uy)
	vLen := math.Sqrt(vx*vx + vy*vy)

	if uLen == 0 || vLen == 0 {
		return 1.0
	}

	dot := (ux*vx + uy*vy) / (uLen * vLen)
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	mult := 0.7 + 0.3*dot
	if mult < 0.6 {
		mult = 0.6
	}
	if mult > 1.0 {
		mult = 1.0
	}
	return mult
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

		st1ID := line.Stations[tr.Segment]
		var nextSegIdx int
		if line.IsLoop {
			n := len(line.Stations)
			nextSegIdx = (tr.Segment + tr.Direction + n) % n
		} else {
			nextSegIdx = tr.Segment + tr.Direction
			if nextSegIdx < 0 {
				nextSegIdx = 0
			}
			if nextSegIdx >= len(line.Stations) {
				nextSegIdx = len(line.Stations) - 1
			}
		}
		st2ID := line.Stations[nextSegIdx]
		segLen := distance(s.State.Stations[st1ID].Pos, s.State.Stations[st2ID].Pos)
		if segLen <= 0 {
			segLen = 10.0
		}

		prof := velocityProfile(tr.Progress)
		cornerMult := trackCornerMultiplier(&s.State, line, tr.Segment, tr.Direction)
		effSpeed := trainSpeed * prof * cornerMult

		progressDelta := (effSpeed * 10.0 / segLen) * dt
		tr.Progress += progressDelta

		// Reached next station
		for tr.Progress >= 1.0 {
			tr.Progress -= 1.0

			if line.IsLoop {
				n := len(line.Stations)
				tr.Segment = (tr.Segment + tr.Direction + n) % n
			} else {
				tr.Segment += tr.Direction

				last := len(line.Stations) - 1

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
		remainingPassengers := make([]Passenger, 0, len(tr.Passengers))
		for _, p := range tr.Passengers {
			if p.Destination == st.Kind {
				s.State.Score++
			} else {
				route := FindOptimalRoute(&s.State.Graph, &s.State, stationID, p.Destination)
				if route.Reachable && route.NextLineID == tr.LineID && (route.NextDirection == 0 || route.NextDirection == tr.Direction) {
					remainingPassengers = append(remainingPassengers, p)
				} else {
					st.Queue = append(st.Queue, p)
				}
			}
		}
		tr.Passengers = remainingPassengers

		// Board
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

		remaining := make([]Passenger, 0, len(st.Queue))
		for _, p := range st.Queue {
			route := getRoute(p.Destination)
			canBoard := len(tr.Passengers) < totalCapacity &&
				route.Reachable &&
				route.NextLineID == tr.LineID &&
				(route.NextDirection == 0 || route.NextDirection == tr.Direction)
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
