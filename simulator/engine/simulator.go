package engine

import (
	"errors"
	"math/rand"
)

type Simulator struct {
	State             GameState
	graphVersion      uint64 // tracks which TopologyVersion the cached Graph was built for
	rng               *rand.Rand
	loopTogglePenalty float64 // P1-4: penalty on rapid loop reversals (<60s)
}

func (s *Simulator) RNG() *rand.Rand {
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(42))
	}
	return s.rng
}

func (s *Simulator) SetSeed(seed uint64) {
	s.rng = rand.New(rand.NewSource(int64(seed)))
}

func NewSimulatorWithWater(stations []Station, rivers []RiverSegment, polygons []WaterPolygon, seed ...uint64) *Simulator {
	for i := range stations {
		stations[i].Alive = true
		stations[i].OvercrowdingTimer = -1
		if stations[i].Capacity == 0 {
			stations[i].Capacity = 6
		}
	}
	var sSeed uint64 = 42
	if len(seed) > 0 {
		sSeed = seed[0]
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
		rng: rand.New(rand.NewSource(int64(sSeed))),
	}
	sim.State.Scheduler.Schedule(rewardInterval(), EventReward)
	sim.State.Scheduler.Schedule(initialSpawnInterval(), EventSpawnStation)
	return sim
}

func NewSimulatorWithRivers(stations []Station, rivers []RiverSegment, seed ...uint64) *Simulator {
	return NewSimulatorWithWater(stations, rivers, nil, seed...)
}

func NewSimulator(stations []Station, seed ...uint64) *Simulator {
	defaultRivers := []RiverSegment{
		{From: Pos{X: 0, Y: 50}, To: Pos{X: 100, Y: 50}, Width: 4.0},
	}
	return NewSimulatorWithWater(stations, defaultRivers, nil, seed...)
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
	s.boardAndAlight(dt)
	s.updateScore()
	s.State.GameTimeSeconds += dt
	s.State.Tick++

	for _, ev := range s.State.Scheduler.Poll(s.State.Tick) {
		switch ev.Kind {
		case EventReward:
			s.offerReward()
		case EventSpawnStation:
			s.spawnStation()
		}
	}

	s.checkGameOver(dt)
}

func (s *Simulator) offerReward() {
	// BUG-6 guard: never overwrite an outstanding reward choice.
	if len(s.State.PendingRewardChoices) > 0 {
		return
	}
	// Weekly reward: always grant one locomotive, then offer choice of one upgrade.
	s.State.Resources.Grant(RewardTrain)
	pool := []RewardType{RewardLine, RewardCarriage, RewardTunnel, RewardTunnel, RewardInterchange}
	s.RNG().Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
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
	case InsertStation:
		return s.insertStation(v)
	case AddTrain:
		return s.addTrain(v)
	case RemoveLine:
		return s.removeLine(v)
	case ChooseReward:
		return s.chooseReward(v)
	case AddCarriage:
		return s.addCarriage(v)
	case RemoveCarriage:
		return s.removeCarriage(v)
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
	if !s.State.Resources.CanSpend(RewardTrain) {
		return errors.New("cannot create a line without an available train")
	}

	// BUG-2: full uniqueness check — detect duplicate stations anywhere in the slice,
	// not just in adjacent pairs (e.g. [0,1,0] would have been accepted before).
	seen := make(map[int]struct{}, len(a.Stations))
	for i, stID := range a.Stations {
		if stID < 0 || stID >= len(s.State.Stations) {
			return errors.New("invalid station ID in line")
		}
		if !s.State.Stations[stID].Alive {
			return errors.New("station is not alive")
		}
		if _, dup := seen[stID]; dup {
			return errors.New("duplicate station in line")
		}
		seen[stID] = struct{}{}
		_ = i // suppress unused-variable warning; index used implicitly
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

	id := -1
	for i, l := range s.State.Lines {
		if l.Removed {
			id = i
			break
		}
	}

	if id == -1 {
		id = len(s.State.Lines)
		s.State.Lines = append(s.State.Lines, Line{
			ID:       id,
			Stations: append([]int(nil), a.Stations...),
			TunnelAt: tunnelAt,
			Removed:  false,
		})
	} else {
		s.State.Lines[id] = Line{
			ID:       id,
			Stations: append([]int(nil), a.Stations...),
			TunnelAt: tunnelAt,
			Removed:  false,
		}
	}

	// Auto-spawn initial train if train resource pool has available trains
	if s.State.Resources.CanSpend(RewardTrain) {
		s.State.Resources.Spend(RewardTrain)
		s.spawnOrCreateTrain(id)
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

func (s *Simulator) spawnOrCreateTrain(lineID int) {
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if !tr.Active {
			tr.LineID = lineID
			tr.Segment = 0
			tr.Progress = 0
			tr.Direction = 1
			tr.Capacity = 6
			tr.Carriages = 1
			tr.Passengers = nil
			tr.Active = true
			tr.JustArrived = true
			tr.JustDeparted = true
			tr.DwellRemaining = 0
			tr.ServiceElapsed = 0
			return
		}
	}

	trID := len(s.State.Trains)
	s.State.Trains = append(s.State.Trains, Train{
		ID:             trID,
		LineID:         lineID,
		Segment:        0,
		Progress:       0,
		Direction:      1,
		Capacity:       6,
		Carriages:      1,
		Active:         true,
		JustArrived:    true,
		JustDeparted:   true,
		DwellRemaining: 0,
		ServiceElapsed: 0,
	})
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

	s.spawnOrCreateTrain(a.LineID)
	return nil
}

func (s *Simulator) addCarriage(a AddCarriage) error {
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID for AddCarriage")
	}
	if s.State.Lines[a.LineID].Removed {
		return errors.New("line is removed")
	}

	// PHASE-4: find any active train on the given line (agent targets line, not slot)
	trID := -1
	for i := range s.State.Trains {
		if s.State.Trains[i].Active && s.State.Trains[i].LineID == a.LineID {
			trID = i
			break
		}
	}
	if trID < 0 {
		return errors.New("no active train on line for AddCarriage")
	}

	if !s.State.Resources.Spend(RewardCarriage) {
		return errors.New("no carriages available")
	}

	s.State.Trains[trID].Carriages++
	return nil
}

func (s *Simulator) removeCarriage(a RemoveCarriage) error {
	if a.TrainID < 0 || a.TrainID >= len(s.State.Trains) {
		return errors.New("invalid train ID")
	}

	tr := &s.State.Trains[a.TrainID]
	if !tr.Active {
		return errors.New("train is inactive")
	}

	if tr.Carriages <= 1 {
		return errors.New("train has no extra carriages to remove")
	}

	tr.Carriages--
	s.State.Resources.Grant(RewardCarriage)
	return nil
}

func (s *Simulator) removeLine(a RemoveLine) error {
	return errors.New("rule violation: lines cannot be removed or shortened")
}

func (s *Simulator) chooseReward(a ChooseReward) error {
	if len(s.State.PendingRewardChoices) == 0 {
		return errors.New("no pending reward choice available")
	}

	// BUG-1: the API contract is strictly positional — a.Choice is always 0 (first card)
	// or 1 (second card). Using the enum value directly would collide because RewardType
	// values 0 and 1 are valid positional indices. The frontend always sends positional
	// indices; any RL / API caller must do the same.
	idx := int(a.Choice)
	if idx < 0 || idx >= len(s.State.PendingRewardChoices) {
		return errors.New("invalid reward choice: expected positional index 0 or 1")
	}
	chosenType := s.State.PendingRewardChoices[idx]

	// Tunnel reward grants two tokens (as per original Mini Metro bonus mechanics).
	if chosenType == RewardTunnel {
		s.State.Resources.Grant(RewardTunnel)
		s.State.Resources.Grant(RewardTunnel)
	} else {
		s.State.Resources.Grant(chosenType)
	}
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
	if len(line.Stations) < 3 {
		return errors.New("line needs at least 3 stations to form a loop")
	}
	firstPos := s.State.Stations[line.Stations[0]].Pos
	lastPos := s.State.Stations[line.Stations[len(line.Stations)-1]].Pos
	needsTunnel := CrossesWater(lastPos, firstPos, s.State.Rivers, s.State.WaterPolygons)

	if needsTunnel {
		if !a.UseTunnel && s.State.Resources.CanSpend(RewardTunnel) {
			a.UseTunnel = true
		}
		if !a.UseTunnel {
			return errors.New("tunnel token required for this wrap-around segment")
		}
		if !s.State.Resources.Spend(RewardTunnel) {
			return errors.New("no tunnel tokens available")
		}
	}

	// P1-4: Apply penalty (-0.50) if loop status is toggled back within 60 seconds (1800 ticks)
	if line.HasBeenLoopToggled && s.State.Tick < line.LastLoopToggleTick+1800 {
		s.loopTogglePenalty += 0.50
	}
	line.IsLoop = true
	line.LoopTunnel = a.UseTunnel
	line.HasBeenLoopToggled = true
	line.LastLoopToggleTick = s.State.Tick
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
	// P1-4: Apply penalty (-0.50) if loop status is toggled back within 60 seconds (1800 ticks)
	if line.HasBeenLoopToggled && s.State.Tick < line.LastLoopToggleTick+1800 {
		s.loopTogglePenalty += 0.50
	}
	if line.LoopTunnel {
		s.State.Resources.Grant(RewardTunnel)
	}
	line.IsLoop = false
	line.LoopTunnel = false
	line.HasBeenLoopToggled = true
	line.LastLoopToggleTick = s.State.Tick
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

	targetLineID := tr.LineID
	if a.LineID >= 0 && a.LineID < len(s.State.Lines) {
		targetLineID = a.LineID
	}

	line := &s.State.Lines[targetLineID]
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

	tr.LineID = targetLineID
	tr.Segment = a.Segment
	tr.Progress = 0
	tr.Direction = dir
	tr.JustArrived = true
	tr.JustDeparted = true
	tr.DwellRemaining = 0

	return nil
}

func (s *Simulator) insertStation(a InsertStation) error {
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

	n := len(line.Stations)
	if a.Index < 1 || a.Index >= n {
		a.Index = n - 1
	}

	stPrev := s.State.Stations[line.Stations[a.Index-1]].Pos
	stNext := s.State.Stations[line.Stations[a.Index]].Pos
	stNew := s.State.Stations[a.StationID].Pos

	tunnelsNeeded := 0
	cross1 := CrossesWater(stPrev, stNew, s.State.Rivers, s.State.WaterPolygons)
	cross2 := CrossesWater(stNew, stNext, s.State.Rivers, s.State.WaterPolygons)
	if cross1 {
		tunnelsNeeded++
	}
	if cross2 {
		tunnelsNeeded++
	}

	origTunnel := false
	if a.Index-1 < len(line.TunnelAt) && line.TunnelAt[a.Index-1] {
		origTunnel = true
	}

	netTunnels := tunnelsNeeded
	if origTunnel {
		netTunnels--
	}

	if netTunnels > 0 {
		if s.State.Resources.Tunnels < netTunnels {
			return errors.New("no tunnel tokens available")
		}
		for i := 0; i < netTunnels; i++ {
			s.State.Resources.Spend(RewardTunnel)
		}
	} else if netTunnels < 0 {
		s.State.Resources.Grant(RewardTunnel)
	}

	// Insert station ID into line.Stations at Index without buffer aliasing
	newStations := make([]int, 0, len(line.Stations)+1)
	newStations = append(newStations, line.Stations[:a.Index]...)
	newStations = append(newStations, a.StationID)
	newStations = append(newStations, line.Stations[a.Index:]...)
	line.Stations = newStations

	// Update tunnel flags without buffer aliasing
	if a.Index-1 < len(line.TunnelAt) {
		newTunnelAt := make([]bool, 0, len(line.TunnelAt)+1)
		newTunnelAt = append(newTunnelAt, line.TunnelAt[:a.Index-1]...)
		newTunnelAt = append(newTunnelAt, cross1, cross2)
		newTunnelAt = append(newTunnelAt, line.TunnelAt[a.Index:]...)
		line.TunnelAt = newTunnelAt
	} else {
		line.TunnelAt = append(line.TunnelAt, cross1)
	}

	// Update active trains running on this line
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if tr.LineID == a.LineID && tr.Active {
			if tr.Segment >= a.Index {
				tr.Segment++
			}
		}
	}

	s.State.TopologyVersion++
	return nil
}

type SimInfo struct {
	EventTriggered string // "none", "reward_offered", "station_spawned", "game_over"
	StepTicks      int
}

// StepMacro applies an action and advances physics for up to duration seconds (in fixed 30 Hz sub-ticks)
// or until an asynchronous event (station spawn, reward choice, game over) triggers.
func (s *Simulator) StepMacro(action Action, duration float64) (obs Observation, reward float64, done bool, info SimInfo) {
	info.EventTriggered = "none"
	if action != nil {
		_ = s.ApplyAction(action)
	}

	if duration <= 0 {
		duration = 5.0
	}
	dt := 1.0 / 30.0
	subTicks := int(duration / dt)
	if subTicks <= 0 {
		subTicks = 1
	}

	startStationCount := len(s.State.Stations)
	initialScore := s.State.Score

	for i := 0; i < subTicks; i++ {
		if !s.State.Alive {
			info.EventTriggered = "game_over"
			break
		}
		if len(s.State.PendingRewardChoices) > 0 {
			info.EventTriggered = "reward_offered"
			break
		}

		s.Step(dt)
		info.StepTicks++

		if !s.State.Alive {
			info.EventTriggered = "game_over"
			break
		}
		if len(s.State.PendingRewardChoices) > 0 {
			info.EventTriggered = "reward_offered"
			break
		}
		if len(s.State.Stations) > startStationCount {
			info.EventTriggered = "station_spawned"
			break
		}
	}

	stepReward := s.ComputeStepReward(s.State.Score - initialScore)

	return s.Observation(), stepReward, !s.State.Alive, info
}
