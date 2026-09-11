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
	disruptionPenalty float64 // operational disruption penalty on structural edits
	ScoringConfig     ScoringConfig // P2-1: configurable reward coefficients and modes
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
		rng:           rand.New(rand.NewSource(int64(sSeed))),
		ScoringConfig: DefaultScoringConfig(),
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

	// Update smoothed queue growth rates every 30 ticks (1 simulation second)
	if s.State.Tick%30 == 0 {
		for i := range s.State.Stations {
			st := &s.State.Stations[i]
			if st.Alive {
				curLen := len(st.Queue)
				delta := float64(curLen - st.PrevQueueLen)
				st.QueueGrowthRate = 0.7*st.QueueGrowthRate + 0.3*delta
				st.PrevQueueLen = curLen
			}
		}
	}

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

	activeLines := 0
	for _, l := range s.State.Lines {
		if !l.Removed {
			activeLines++
		}
	}
	unlockedLines := activeLines + s.State.Resources.Lines

	var pool []RewardType
	if unlockedLines < MaxLines {
		pool = []RewardType{RewardLine, RewardCarriage, RewardTunnel, RewardTunnel, RewardInterchange}
	} else {
		pool = []RewardType{RewardCarriage, RewardTunnel, RewardTunnel, RewardInterchange}
	}

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
			ID:               id,
			Stations:         append([]int(nil), a.Stations...),
			TunnelAt:         tunnelAt,
			Removed:          false,
			LastModifiedTick: s.State.Tick,
		})
	} else {
		s.State.Lines[id] = Line{
			ID:               id,
			Stations:         append([]int(nil), a.Stations...),
			TunnelAt:         tunnelAt,
			Removed:          false,
			LastModifiedTick: s.State.Tick,
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

	line.LastModifiedTick = s.State.Tick
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
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}

	line := &s.State.Lines[a.LineID]
	if line.Removed || len(line.Stations) < 2 {
		return errors.New("line is already removed or invalid")
	}

	// Assess calibrated operational disruption penalty before clearing line state
	dumpedPax := 0
	activeTrainsCount := 0
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if tr.Active && tr.LineID == a.LineID {
			activeTrainsCount++
			dumpedPax += len(tr.Passengers)
		}
	}

	N := len(s.State.Stations)
	linesPerStation := make([]int, N)
	for _, l := range s.State.Lines {
		if l.Removed {
			continue
		}
		seen := make(map[int]bool)
		for _, stID := range l.Stations {
			if stID >= 0 && stID < N && !seen[stID] {
				linesPerStation[stID]++
				seen[stID] = true
			}
		}
	}
	severedStations := 0
	for _, stID := range line.Stations {
		if stID >= 0 && stID < N && linesPerStation[stID] <= 1 {
			severedStations++
		}
	}

	disruption := 1.0 + 0.20*float64(dumpedPax) + 0.50*float64(activeTrainsCount) + 1.50*float64(severedStations)
	s.disruptionPenalty += disruption

	// 1. Refund tunnel tokens used by the line segments
	for _, isTunnel := range line.TunnelAt {
		if isTunnel {
			s.State.Resources.Grant(RewardTunnel)
		}
	}

	if line.IsLoop && line.LoopTunnel {
		s.State.Resources.Grant(RewardTunnel)
	}

	// 2. Refund the line token
	s.State.Resources.Grant(RewardLine)

	// 3. Process active trains on this line
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if tr.Active && tr.LineID == a.LineID {
			// Refund extra carriages
			for c := 1; c < tr.Carriages; c++ {
				s.State.Resources.Grant(RewardCarriage)
			}
			// Refund the train asset
			s.State.Resources.Grant(RewardTrain)

			// Disembark passengers to nearest station
			if len(tr.Passengers) > 0 {
				nearestStID := -1
				n := len(line.Stations)
				if n == 1 {
					nearestStID = line.Stations[0]
				} else if n >= 2 {
					seg := tr.Segment
					var nextSeg int
					if line.IsLoop {
						seg = ((seg % n) + n) % n
						nextSeg = (seg + 1) % n
					} else {
						if seg < 0 {
							seg = 0
						}
						if seg >= n-1 {
							seg = n - 2
						}
						nextSeg = seg + 1
					}

					if tr.Progress < 0.5 {
						nearestStID = line.Stations[seg]
					} else {
						nearestStID = line.Stations[nextSeg]
					}
				}

				if nearestStID >= 0 && nearestStID < len(s.State.Stations) {
					st := &s.State.Stations[nearestStID]
					for _, p := range tr.Passengers {
						if st.Alive {
							st.Queue = append(st.Queue, p)
						}
					}
				}
			}

			// Deactivate and reset train
			tr.Active = false
			tr.LineID = -1
			tr.Segment = 0
			tr.Progress = 0
			tr.Direction = 1
			tr.Capacity = 6
			tr.Carriages = 1
			tr.Passengers = nil
			tr.JustArrived = false
			tr.JustDeparted = false
			tr.DwellRemaining = 0
			tr.ServiceElapsed = 0
		}
	}

	// 4. Mark line removed and clear its properties
	line.Removed = true
	line.Stations = nil
	line.TunnelAt = nil
	line.IsLoop = false
	line.LoopTunnel = false
	line.LastLoopToggleTick = 0
	line.HasBeenLoopToggled = false
	line.LastModifiedTick = s.State.Tick

	// 5. Invalidate network topology
	s.State.TopologyVersion++
	return nil
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
	if a.LineID < 0 || a.LineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}

	line := &s.State.Lines[a.LineID]
	if line.Removed {
		return errors.New("line is removed")
	}
	if line.IsLoop {
		return errors.New("cannot shorten a loop line; open loop first")
	}
	if len(line.Stations) <= 2 {
		return errors.New("cannot shorten line with 2 or fewer stations; use removeLine instead")
	}

	if a.FromFront {
		// Remove front station (line.Stations[0])
		if len(line.TunnelAt) > 0 && line.TunnelAt[0] {
			s.State.Resources.Grant(RewardTunnel)
		}
		if len(line.TunnelAt) > 0 {
			line.TunnelAt = line.TunnelAt[1:]
		}
		line.Stations = line.Stations[1:]

		// Update trains on this line
		for i := range s.State.Trains {
			tr := &s.State.Trains[i]
			if tr.Active && tr.LineID == a.LineID {
				if tr.Segment == 0 {
					// Train was on the removed front segment; place at new front station
					tr.Segment = 0
					tr.Progress = 0.0
					tr.Direction = 1
					tr.JustArrived = true
					tr.JustDeparted = true
					tr.DwellRemaining = dwellTime
				} else {
					tr.Segment--
				}
			}
		}
	} else {
		// Remove tail station (line.Stations[len(line.Stations)-1])
		lastTunnelIdx := len(line.TunnelAt) - 1
		if lastTunnelIdx >= 0 && line.TunnelAt[lastTunnelIdx] {
			s.State.Resources.Grant(RewardTunnel)
		}
		if lastTunnelIdx >= 0 {
			line.TunnelAt = line.TunnelAt[:lastTunnelIdx]
		}
		line.Stations = line.Stations[:len(line.Stations)-1]

		newTail := len(line.Stations) - 1
		// Update trains on this line
		for i := range s.State.Trains {
			tr := &s.State.Trains[i]
			if tr.Active && tr.LineID == a.LineID {
				if tr.Segment >= newTail {
					// Train was on the removed tail segment; place at new tail station
					tr.Segment = newTail
					tr.Progress = 0.0
					tr.Direction = -1
					tr.JustArrived = true
					tr.JustDeparted = true
					tr.DwellRemaining = dwellTime
				}
			}
		}
	}

	s.disruptionPenalty += 0.10 // small local modification cost
	line.LastModifiedTick = s.State.Tick
	s.State.TopologyVersion++
	s.rebuildGraphIfNeeded()

	// Disembark passengers on this line who can no longer reach their destination
	// on this line or whose route required the removed endpoint station.
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if !tr.Active || tr.LineID != a.LineID || len(tr.Passengers) == 0 {
			continue
		}

		curStationIndex := tr.Segment
		if tr.Progress >= 0.5 {
			nextSeg := tr.Segment + tr.Direction
			if nextSeg >= 0 && nextSeg < len(line.Stations) {
				curStationIndex = nextSeg
			}
		}
		if curStationIndex < 0 {
			curStationIndex = 0
		} else if curStationIndex >= len(line.Stations) {
			curStationIndex = len(line.Stations) - 1
		}
		curStationID := line.Stations[curStationIndex]

		var retainedPassengers []Passenger
		for _, p := range tr.Passengers {
			route := FindOptimalRoute(&s.State.Graph, &s.State, curStationID, p.Destination)
			if !route.Reachable || route.NextLineID != tr.LineID || (route.NextDirection != 0 && route.NextDirection != tr.Direction) {
				if curStationID >= 0 && curStationID < len(s.State.Stations) {
					st := &s.State.Stations[curStationID]
					if st.Alive {
						st.Queue = append(st.Queue, p)
					}
				}
			} else {
				retainedPassengers = append(retainedPassengers, p)
			}
		}
		tr.Passengers = retainedPassengers
	}

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

// ReverseLine reverses the station sequence of an active non-loop line,
// preserving identical physical connectivity and updating active trains.
func (s *Simulator) ReverseLine(lineID int) error {
	if lineID < 0 || lineID >= len(s.State.Lines) {
		return errors.New("invalid line ID")
	}
	line := &s.State.Lines[lineID]
	if line.Removed || len(line.Stations) < 2 || line.IsLoop {
		return nil
	}

	n := len(line.Stations)
	// 1. Reverse Stations slice
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		line.Stations[i], line.Stations[j] = line.Stations[j], line.Stations[i]
	}

	// 2. Reverse TunnelAt slice (length is n-1)
	nTunnels := len(line.TunnelAt)
	for i, j := 0, nTunnels-1; i < j; i, j = i+1, j-1 {
		line.TunnelAt[i], line.TunnelAt[j] = line.TunnelAt[j], line.TunnelAt[i]
	}

	// 3. Update active trains on this line
	for i := range s.State.Trains {
		tr := &s.State.Trains[i]
		if !tr.Active || tr.LineID != lineID {
			continue
		}
		// Segment k in original line corresponds to (n - 1) - k in reversed line
		tr.Segment = (n - 1) - tr.Segment
		if tr.Segment < 0 {
			tr.Segment = 0
		} else if tr.Segment >= n {
			tr.Segment = n - 1
		}
		tr.Direction = -tr.Direction
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

	line.LastModifiedTick = s.State.Tick
	s.State.TopologyVersion++
	return nil
}


type SimInfo struct {
	EventTriggered string // "none", "reward_offered", "station_spawned", "game_over"
	StepTicks      int
}

// StepMacroBreakdown applies an action and advances physics for up to duration seconds (in fixed 30 Hz sub-ticks)
// or until an asynchronous event (station spawn, reward choice, game over) triggers, returning the decomposed reward breakdown (P2-1).
func (s *Simulator) StepMacroBreakdown(action Action, duration float64) (obs Observation, reward float64, done bool, info SimInfo, rb RewardBreakdown) {
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

	stepReward, breakdown := s.ComputeStepRewardBreakdown(s.State.Score - initialScore)

	return s.Observation(), stepReward, !s.State.Alive, info, breakdown
}

// StepMacro applies an action and advances physics for up to duration seconds (in fixed 30 Hz sub-ticks)
// or until an asynchronous event (station spawn, reward choice, game over) triggers.
func (s *Simulator) StepMacro(action Action, duration float64) (obs Observation, reward float64, done bool, info SimInfo) {
	obs, reward, done, info, _ = s.StepMacroBreakdown(action, duration)
	return obs, reward, done, info
}

// Clone produces an independent deep copy of the Simulator, preserving complete state
// (stations with queues, lines with stations/tunnels, trains with passengers, scheduler events,
// resources, topology graph, and RNG) for fast counterfactual rollouts without mutating the original.
func (s *Simulator) Clone() *Simulator {
	stationsCopy := make([]Station, len(s.State.Stations))
	for i := range s.State.Stations {
		stationsCopy[i] = s.State.Stations[i]
		if len(s.State.Stations[i].Queue) > 0 {
			stationsCopy[i].Queue = make([]Passenger, len(s.State.Stations[i].Queue))
			copy(stationsCopy[i].Queue, s.State.Stations[i].Queue)
		} else {
			stationsCopy[i].Queue = nil
		}
	}

	linesCopy := make([]Line, len(s.State.Lines))
	for i := range s.State.Lines {
		linesCopy[i] = s.State.Lines[i]
		if len(s.State.Lines[i].Stations) > 0 {
			linesCopy[i].Stations = make([]int, len(s.State.Lines[i].Stations))
			copy(linesCopy[i].Stations, s.State.Lines[i].Stations)
		}
		if len(s.State.Lines[i].TunnelAt) > 0 {
			linesCopy[i].TunnelAt = make([]bool, len(s.State.Lines[i].TunnelAt))
			copy(linesCopy[i].TunnelAt, s.State.Lines[i].TunnelAt)
		}
	}

	trainsCopy := make([]Train, len(s.State.Trains))
	for i := range s.State.Trains {
		trainsCopy[i] = s.State.Trains[i]
		if len(s.State.Trains[i].Passengers) > 0 {
			trainsCopy[i].Passengers = make([]Passenger, len(s.State.Trains[i].Passengers))
			copy(trainsCopy[i].Passengers, s.State.Trains[i].Passengers)
		} else {
			trainsCopy[i].Passengers = nil
		}
	}

	riversCopy := append([]RiverSegment(nil), s.State.Rivers...)

	waterPolysCopy := make([]WaterPolygon, len(s.State.WaterPolygons))
	for i := range s.State.WaterPolygons {
		waterPolysCopy[i].Vertices = append([]Pos(nil), s.State.WaterPolygons[i].Vertices...)
	}


	schedulerCopy := EventScheduler{
		Events: append([]ScheduledEvent(nil), s.State.Scheduler.Events...),
	}

	rewardsCopy := append([]RewardType(nil), s.State.PendingRewardChoices...)

	var weightsCopy map[StationKind]int
	if s.State.StationWeights != nil {
		weightsCopy = make(map[StationKind]int, len(s.State.StationWeights))
		for k, v := range s.State.StationWeights {
			weightsCopy[k] = v
		}
	}

	graphCopy := NetworkGraph{
		Adj:       make(map[int][]int, len(s.State.Graph.Adj)),
		EdgeLines: make(map[[2]int][]int, len(s.State.Graph.EdgeLines)),
	}
	for k, v := range s.State.Graph.Adj {
		graphCopy.Adj[k] = append([]int(nil), v...)
	}
	for k, v := range s.State.Graph.EdgeLines {
		graphCopy.EdgeLines[k] = append([]int(nil), v...)
	}

	rngCopy := rand.New(rand.NewSource(int64(s.State.Tick)*10007 + int64(s.State.Score) + 42))

	return &Simulator{
		State: GameState{
			MapName:              s.State.MapName,
			Stations:             stationsCopy,
			Lines:                linesCopy,
			Trains:               trainsCopy,
			Rivers:               riversCopy,
			WaterPolygons:        waterPolysCopy,
			Resources:            s.State.Resources,
			Graph:                graphCopy,
			Scheduler:            schedulerCopy,
			PendingRewardChoices: rewardsCopy,
			TopologyVersion:      s.State.TopologyVersion,
			NextPassengerID:      s.State.NextPassengerID,
			Score:                s.State.Score,
			Tick:                 s.State.Tick,
			GameTimeSeconds:      s.State.GameTimeSeconds,
			Alive:                s.State.Alive,
			MaxTrainsPerLine:     s.State.MaxTrainsPerLine,
			StationWeights:       weightsCopy,
		},
		graphVersion:      s.graphVersion,
		rng:               rngCopy,
		loopTogglePenalty: s.loopTogglePenalty,
		disruptionPenalty: s.disruptionPenalty,
		ScoringConfig:     s.ScoringConfig,
	}
}

