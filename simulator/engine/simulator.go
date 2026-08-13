package engine

import (
	"errors"
	"math/rand"
)

type Simulator struct {
	State        GameState
	graphVersion uint64 // tracks which TopologyVersion the cached Graph was built for
}

func NewSimulatorWithWater(stations []Station, rivers []RiverSegment, polygons []WaterPolygon) *Simulator {
	for i := range stations {
		stations[i].Alive = true
		stations[i].OvercrowdingTimer = -1
		if stations[i].Capacity == 0 {
			stations[i].Capacity = 6
		}
	}
	sim := &Simulator{
		State: GameState{
			Stations:         stations,
			Lines:            []Line{},
			Trains:           []Train{},
			Rivers:           rivers,
			WaterPolygons:    polygons,
			Resources:        NewResourcePool(),
			Score:            0,
			Tick:             0,
			Alive:            true,
			MaxTrainsPerLine: 4,
		},
	}
	sim.State.Scheduler.Schedule(rewardInterval(), EventReward)
	sim.State.Scheduler.Schedule(spawnInterval(), EventSpawnStation)
	return sim
}

func NewSimulatorWithRivers(stations []Station, rivers []RiverSegment) *Simulator {
	return NewSimulatorWithWater(stations, rivers, nil)
}

func NewSimulator(stations []Station) *Simulator {
	defaultRivers := []RiverSegment{
		{From: Pos{X: 0, Y: 50}, To: Pos{X: 100, Y: 50}, Width: 4.0},
	}
	return NewSimulatorWithWater(stations, defaultRivers, nil)
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
	case CloseLoop:
		return s.closeLoop(v)
	case OpenLoop:
		return s.openLoop(v)
	case RepositionTrain:
		return s.repositionTrain(v)
	default:
		return errors.New("unknown action type")
	}
}

func (s *Simulator) addLine(a AddLine) error {
	if len(a.Stations) < 2 {
		return errors.New("insufficient stations to add a new line")
	}

	for i, stID := range a.Stations {
		if stID < 0 || stID >= len(s.State.Stations) {
			return errors.New("invalid station ID in line")
		}
		if !s.State.Stations[stID].Alive {
			return errors.New("station is not alive")
		}
		if i+1 < len(a.Stations) && a.Stations[i] == a.Stations[i+1] {
			return errors.New("cannot connect station to itself")
		}
	}

	tunnelAt := make([]bool, len(a.Stations)-1)
	tunnelsNeeded := 0
	for i := 0; i+1 < len(a.Stations); i++ {
		uPos := s.State.Stations[a.Stations[i]].Pos
		vPos := s.State.Stations[a.Stations[i+1]].Pos
		if CrossesWater(uPos, vPos, s.State.Rivers, s.State.WaterPolygons) {
			tunnelAt[i] = true
			tunnelsNeeded++
		}
	}

	if tunnelsNeeded > 0 && s.State.Resources.Tunnels < tunnelsNeeded {
		return errors.New("no tunnel tokens available")
	}

	if !s.State.Resources.Spend(RewardLine) {
		return errors.New("no lines available")
	}

	for i := 0; i < tunnelsNeeded; i++ {
		s.State.Resources.Spend(RewardTunnel)
	}

	id := len(s.State.Lines)

	s.State.Lines = append(s.State.Lines, Line{
		ID:       id,
		Stations: append([]int(nil), a.Stations...),
		TunnelAt: tunnelAt,
		Removed:  false,
	})

	// Auto-spawn initial train if train resource pool has available trains
	if s.State.Resources.CanSpend(RewardTrain) {
		s.State.Resources.Spend(RewardTrain)
		trID := len(s.State.Trains)
		s.State.Trains = append(s.State.Trains, Train{
			ID:          trID,
			LineID:      id,
			Segment:     0,
			Progress:    0,
			Direction:   1,
			Capacity:    6,
			Carriages:   1,
			Active:      true,
			JustArrived: true,
		})
	}

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

	if !s.State.Stations[a.StationID].Alive {
		return errors.New("station is not alive")
	}

	line := &s.State.Lines[a.LineID]

	if line.Removed {
		return errors.New("line is removed")
	}

	for _, stID := range line.Stations {
		if stID == a.StationID {
			return errors.New("station is already on this line")
		}
	}

	var endpointID int
	if a.FromFront {
		endpointID = line.Stations[0]
	} else {
		endpointID = line.Stations[len(line.Stations)-1]
	}

	endpointPos := s.State.Stations[endpointID].Pos
	newPos := s.State.Stations[a.StationID].Pos
	needsTunnel := CrossesWater(endpointPos, newPos, s.State.Rivers, s.State.WaterPolygons)

	if needsTunnel {
		if !a.UseTunnel && s.State.Resources.CanSpend(RewardTunnel) {
			a.UseTunnel = true
		}
		if !a.UseTunnel {
			return errors.New("tunnel token required for this segment")
		}
		if !s.State.Resources.Spend(RewardTunnel) {
			return errors.New("no tunnel tokens available")
		}
	}

	if a.FromFront {
		line.Stations = append([]int{a.StationID}, line.Stations...)
		line.TunnelAt = append([]bool{a.UseTunnel}, line.TunnelAt...)
		for i := range s.State.Trains {
			tr := &s.State.Trains[i]
			if tr.LineID == a.LineID && tr.Active {
				tr.Segment++
			}
		}
	} else {
		line.Stations = append(line.Stations, a.StationID)
		line.TunnelAt = append(line.TunnelAt, a.UseTunnel)
	}

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

	activeCount := 0
	for _, tr := range s.State.Trains {
		if tr.Active && tr.LineID == a.LineID {
			activeCount++
		}
	}
	maxLimit := s.State.MaxTrainsPerLine
	if maxLimit <= 0 {
		maxLimit = 4
	}
	if activeCount >= maxLimit {
		return errors.New("max trains per line limit reached")
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

	// refund tunnels used by this line
	for _, isTunnel := range line.TunnelAt {
		if isTunnel {
			s.State.Resources.Grant(RewardTunnel)
		}
	}
	if line.IsLoop && line.LoopTunnel {
		s.State.Resources.Grant(RewardTunnel)
	}

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
	if line.IsLoop {
		return errors.New("open the loop before shortening")
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
			if tr.Segment <= 0 {
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

// closeLoop connects the last station back to the first, making the line a one-way loop.
func (s *Simulator) closeLoop(a CloseLoop) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}
	line := &s.State.Lines[a.LineID]
	if line.Removed {
		return errors.New("line is removed")
	}
	if line.IsLoop {
		return errors.New("line is already a loop")
	}
	if len(line.Stations) < 2 {
		return errors.New("line needs at least 2 stations to form a loop")
	}
	firstPos := s.State.Stations[line.Stations[0]].Pos
	lastPos := s.State.Stations[line.Stations[len(line.Stations)-1]].Pos
	needsTunnel := CrossesWater(lastPos, firstPos, s.State.Rivers, s.State.WaterPolygons)
	if needsTunnel && !a.UseTunnel {
		return errors.New("tunnel token required for this wrap-around segment")
	}
	if a.UseTunnel && !needsTunnel {
		return errors.New("tunnel not required for this wrap-around segment")
	}
	if a.UseTunnel {
		if !s.State.Resources.Spend(RewardTunnel) {
			return errors.New("no tunnel tokens available")
		}
	}
	line.IsLoop = true
	line.LoopTunnel = a.UseTunnel
	s.State.TopologyVersion++
	return nil
}

// openLoop breaks the loop back into a linear line and refunds the tunnel token if one was used.
func (s *Simulator) openLoop(a OpenLoop) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}
	line := &s.State.Lines[a.LineID]
	if line.Removed {
		return errors.New("line is removed")
	}
	if !line.IsLoop {
		return errors.New("line is not a loop")
	}
	if line.LoopTunnel {
		s.State.Resources.Grant(RewardTunnel)
	}
	line.IsLoop = false
	line.LoopTunnel = false
	// trains that were heading "through" the wrap-around now need a valid direction
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if !tr.Active || tr.LineID != a.LineID {
			continue
		}
		// normalise: trains already within bounds are fine; just ensure direction is legal
		last := len(line.Stations) - 1
		if tr.Segment >= last {
			tr.Segment = last
			tr.Direction = -1
		} else if tr.Segment <= 0 {
			tr.Segment = 0
			tr.Direction = 1
		}
	}
	s.State.TopologyVersion++
	return nil
}

// repositionTrain moves an active train to a specific station segment on its line.
func (s *Simulator) repositionTrain(a RepositionTrain) error {
	if a.TrainID < 0 || a.TrainID >= len(s.State.Trains) {
		return errors.New("invalid train ID")
	}
	tr := &s.State.Trains[a.TrainID]
	if !tr.Active {
		return errors.New("train is inactive")
	}
	if tr.LineID < 0 || tr.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}
	line := &s.State.Lines[tr.LineID]
	if line.Removed || len(line.Stations) < 2 {
		return errors.New("invalid or removed line")
	}
	if a.Segment < 0 || a.Segment >= len(line.Stations) {
		return errors.New("invalid segment station index")
	}

	dir := a.Direction
	if line.IsLoop {
		dir = 1
	} else {
		if a.Segment == 0 {
			dir = 1
		} else if a.Segment == len(line.Stations)-1 {
			dir = -1
		} else if dir != 1 && dir != -1 {
			dir = 1
		}
	}

	tr.Segment = a.Segment
	tr.Progress = 0
	tr.Direction = dir
	tr.JustArrived = true
	tr.DwellRemaining = 0

	return nil
}

