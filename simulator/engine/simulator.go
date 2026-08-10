package engine

import (
	"errors"
	"math/rand"
)

type Simulator struct {
	State        GameState
	graphVersion uint64 // tracks which TopologyVersion the cached Graph was built for
}

func NewSimulator(stations []Station) *Simulator {
	for i := range stations {
		stations[i].Alive = true
		stations[i].OvercrowdingTimer = -1
		if stations[i].Capacity == 0 {
			stations[i].Capacity = 6
		}
	}
	sim := &Simulator{
		State: GameState{
			Stations:  stations,
			Lines:     []Line{},
			Trains:    []Train{},
			Resources: NewResourcePool(),
			Score:     0,
			Tick:      0,
			Alive:     true,
		},
	}
	sim.State.Scheduler.Schedule(rewardInterval(), EventReward)
	sim.State.Scheduler.Schedule(spawnInterval(), EventSpawnStation)
	return sim
}

// rebuildGraphIfNeeded rebuilds the cached NetworkGraph whenever the network
// topology has changed since the last rebuild (detected via TopologyVersion).
func (s *Simulator) rebuildGraphIfNeeded() {
	if s.graphVersion != s.State.TopologyVersion {
		s.State.Graph = BuildGraph(&s.State)
		s.graphVersion = s.State.TopologyVersion
	}
}

func (s *Simulator) Step(dt float64) {
	if !s.State.Alive {
		return
	}

	// Freeze simulation movement while a reward choice is pending
	if len(s.State.PendingRewardChoices) > 0 {
		return
	}

	s.rebuildGraphIfNeeded()
	s.spawnPassengers(dt)
	s.moveTrains(dt)
	s.boardAndAlight()
	s.updateScore()
	s.State.Tick++

	for _, ev := range s.State.Scheduler.Poll(s.State.Tick) {
		switch ev.Kind {
		case EventReward:
			s.offerReward()
		case EventSpawnStation:
			s.spawnStation()
		}
	}

	s.checkGameOver()
}

func (s *Simulator) offerReward() {
	s.State.Resources.Grant(RewardTrain)
	pool := []RewardType{RewardLine, RewardCarriage, RewardTunnel, RewardInterchange}
	rand.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	s.State.PendingRewardChoices = pool[:2]
	s.State.Scheduler.Schedule(s.State.Tick+rewardInterval(), EventReward)
}

func (s *Simulator) ApplyAction(a Action) error {
	if !s.State.Alive {
		return errors.New("game is over")
	}
	switch v := a.(type) {
	case AddLine:
		return s.addLine(v)
	case ExtendLine:
		return s.extendLine(v)
	case AddTrain:
		return s.addTrain(v)
	case RemoveLine:
		return s.removeLine(v)
	case ChooseReward:
		return s.chooseReward(v)
	case AddCarriage:
		return s.addCarriage(v)
	case UpgradeInterchange:
		return s.upgradeInterchange(v)
	case ShortenLine:
		return s.shortenLine(v)
	default:
		return errors.New("unknown action type")
	}
}

func (s *Simulator) addLine(a AddLine) error {
	if len(a.Stations) < 2 {
		return errors.New("insufficient stations to add a new line")
	}

	for _, stID := range a.Stations {
		if stID < 0 || stID >= len(s.State.Stations) {
			return errors.New("invalid station ID in line")
		}
	}

	if !s.State.Resources.Spend(RewardLine) {
		return errors.New("no lines available")
	}

	id := len(s.State.Lines)
	// initial segments are never tunnel crossings
	tunnelAt := make([]bool, len(a.Stations)-1)

	s.State.Lines = append(s.State.Lines, Line{
		ID:       id,
		Stations: append([]int(nil), a.Stations...),
		TunnelAt: tunnelAt,
		Removed:  false,
	})

	s.State.TopologyVersion++
	return nil
}

func (s *Simulator) extendLine(a ExtendLine) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}

	if a.StationID < 0 || a.StationID >= len(s.State.Stations) {
		return errors.New("invalid station ID")
	}

	line := &s.State.Lines[a.LineID]

	if line.Removed {
		return errors.New("line is removed")
	}

	if len(line.Stations) > 0 && line.Stations[len(line.Stations)-1] == a.StationID {
		return errors.New("cannot extend line to the same station")
	}

	// check whether this segment crosses a tunnel threshold
	lastID := line.Stations[len(line.Stations)-1]
	lastPos := s.State.Stations[lastID].Pos
	newPos := s.State.Stations[a.StationID].Pos
	needsTunnel := distance(lastPos, newPos) > tunnelDistanceThreshold

	if needsTunnel && !a.UseTunnel {
		return errors.New("tunnel token required for this segment")
	}
	if a.UseTunnel && !needsTunnel {
		return errors.New("tunnel not required for this segment")
	}
	if a.UseTunnel {
		if !s.State.Resources.Spend(RewardTunnel) {
			return errors.New("no tunnel tokens available")
		}
	}

	line.Stations = append(line.Stations, a.StationID)
	line.TunnelAt = append(line.TunnelAt, a.UseTunnel)
	s.State.TopologyVersion++
	return nil
}

func (s *Simulator) addTrain(a AddTrain) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}

	line := &s.State.Lines[a.LineID]

	if line.Removed || len(line.Stations) < 2 {
		return errors.New("invalid or removed line")
	}

	if !s.State.Resources.Spend(RewardTrain) {
		return errors.New("no trains available")
	}

	id := len(s.State.Trains)

	s.State.Trains = append(s.State.Trains, Train{
		ID:          id,
		LineID:      a.LineID,
		Segment:     0,
		Progress:    0,
		Direction:   1,
		Capacity:    6,
		Carriages:   1,
		Active:      true,
		JustArrived: true,
	})

	return nil
}

func (s *Simulator) addCarriage(a AddCarriage) error {
	if a.TrainID < 0 || a.TrainID >= len(s.State.Trains) {
		return errors.New("invalid train ID")
	}

	tr := &s.State.Trains[a.TrainID]
	if !tr.Active {
		return errors.New("train is inactive")
	}

	if !s.State.Resources.Spend(RewardCarriage) {
		return errors.New("no carriages available")
	}

	tr.Carriages++
	return nil
}

func (s *Simulator) removeLine(a RemoveLine) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}

	line := &s.State.Lines[a.LineID]
	if line.Removed {
		return errors.New("line is already removed")
	}

	line.Removed = true
	s.State.Resources.Grant(RewardLine)
	s.State.TopologyVersion++

	// deactivate trains in this line and refund train and carriage resources
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if tr.LineID == a.LineID && tr.Active {
			tr.Active = false
			s.State.Resources.Grant(RewardTrain)
			if tr.Carriages > 1 {
				extra := tr.Carriages - 1
				for k := 0; k < extra; k++ {
					s.State.Resources.Grant(RewardCarriage)
				}
				tr.Carriages = 1
			}
		}
	}

	return nil
}

func (s *Simulator) chooseReward(a ChooseReward) error {
	if len(s.State.PendingRewardChoices) == 0 {
		return errors.New("no pending reward choice available")
	}

	valid := false
	for _, choice := range s.State.PendingRewardChoices {
		if choice == a.Choice {
			valid = true
			break
		}
	}

	if !valid {
		return errors.New("invalid reward choice")
	}

	s.State.Resources.Grant(a.Choice)
	s.State.PendingRewardChoices = nil
	return nil
}

// upgradeInterchange spends one interchange token and marks the given station as an interchange hub.
func (s *Simulator) upgradeInterchange(a UpgradeInterchange) error {
	if a.StationID < 0 || a.StationID >= len(s.State.Stations) {
		return errors.New("invalid station ID")
	}
	st := &s.State.Stations[a.StationID]
	if !st.Alive {
		return errors.New("station is not alive")
	}
	if st.IsInterchange {
		return errors.New("station is already an interchange")
	}
	if !s.State.Resources.Spend(RewardInterchange) {
		return errors.New("no interchange tokens available")
	}
	st.IsInterchange = true
	st.Capacity = 18 // real Mini Metro interchange capacity (3× base 6)
	return nil
}

// shortenLine removes one station from either endpoint of a line.
func (s *Simulator) shortenLine(a ShortenLine) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}
	line := &s.State.Lines[a.LineID]
	if line.Removed {
		return errors.New("line is removed")
	}
	if len(line.Stations) <= 2 {
		return errors.New("line must keep at least 2 stations")
	}

	if a.FromFront {
		// refund tunnel token if the first segment (stations[0]→stations[1]) was a tunnel
		if len(line.TunnelAt) > 0 && line.TunnelAt[0] {
			s.State.Resources.Grant(RewardTunnel)
		}
		line.Stations = line.Stations[1:]
		if len(line.TunnelAt) > 0 {
			line.TunnelAt = line.TunnelAt[1:]
		}
		// all train segments shift down by 1 since every station index decreased by 1
		for i := range s.State.Trains {
			tr := &s.State.Trains[i]
			if !tr.Active || tr.LineID != a.LineID {
				continue
			}
			tr.Segment--
			if tr.Segment < 0 {
				tr.Segment = 0
				tr.Progress = 0
				tr.Direction = 1 // was heading toward the now-removed first station; reverse
			}
		}
	} else {
		// refund tunnel token if the last segment (stations[n-2]→stations[n-1]) was a tunnel
		lastSeg := len(line.TunnelAt) - 1
		if lastSeg >= 0 && line.TunnelAt[lastSeg] {
			s.State.Resources.Grant(RewardTunnel)
		}
		line.Stations = line.Stations[:len(line.Stations)-1]
		if len(line.TunnelAt) > 0 {
			line.TunnelAt = line.TunnelAt[:len(line.TunnelAt)-1]
		}
		// clamp trains that were at or past the now-removed last station
		newLast := len(line.Stations) - 1
		for i := range s.State.Trains {
			tr := &s.State.Trains[i]
			if !tr.Active || tr.LineID != a.LineID {
				continue
			}
			if tr.Segment > newLast {
				tr.Segment = newLast
				tr.Progress = 0
				tr.Direction = -1 // bounce back from the new end
			}
		}
	}

	s.State.TopologyVersion++
	return nil
}
