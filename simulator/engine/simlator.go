package engine

type Simulator struct {
	State GameState
}

func (s *Simulator) Step(dt float64) {
	s.spawnPassengers(dt)
	s.moveTrains(dt)
	s.boardAndAlight()
	s.updateScore()
	s.State.Tick++
	s.checkGameOver()
}

func (s *Simulator) ApplyAction(a Action) {
	switch v := a.(type) {

	case AddLine:
		s.addLine(v)

	case ExtendLine:
		s.extendLine(v)

	case AddTrain:
		s.addTrain(v)

	case RemoveLine:
		s.removeLine(v)
	}
}

func (s *Simulator) addLine(a AddLine) {
	if len(a.Stations) < 2 {
		return
	}

	id := len(s.State.Lines)

	s.State.Lines = append(s.State.Lines, Line{
		ID:       id,
		Stations: append([]int(nil), a.Stations...),
		Removed:  false,
	})
}

func (s *Simulator) extendLine(a ExtendLine) {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return
	}

	if a.StationID < 0 || a.StationID >= len(s.State.Stations) {
		return
	}

	line := &s.State.Lines[a.LineID]

	if line.Removed {
		return
	}

	line.Stations = append(line.Stations, a.StationID)
}

func (s *Simulator) addTrain(a AddTrain) {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return
	}

	line := &s.State.Lines[a.LineID]

	if line.Removed {
		return
	}

	id := len(s.State.Trains)

	s.State.Trains = append(s.State.Trains, Train{
		ID:        id,
		LineID:    a.LineID,
		Segment:   0,
		Progress:  0,
		Direction: 1,
		Capacity:  6,
		Active:    true,
	})
}

func (s *Simulator) removeLine(a RemoveLine) {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return
	}

	line := &s.State.Lines[a.LineID]
	line.Removed = true

	// deactive trains in this line
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]

		if tr.LineID == a.LineID {
			tr.Active = false
		}
	}
}
