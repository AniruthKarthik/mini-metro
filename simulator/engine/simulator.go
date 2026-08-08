package engine

import (
	"errors"
	"math/rand"
)

type Simulator struct {
	State GameState
}

func NewSimulator(stations []Station) *Simulator {
	for i := range stations {
		stations[i].Alive = true
	}
	return &Simulator{
		State: GameState{
			Stations:     stations,
			Lines:        []Line{},
			Trains:       []Train{},
			Resources:    NewResourcePool(),
			NextRewardAt: rewardInterval(),
			Score:        0,
			Tick:         0,
			Alive:        true,
		},
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

	s.spawnPassengers(dt)
	s.moveTrains(dt)
	s.boardAndAlight()
	s.updateScore()
	s.State.Tick++

	if float64(s.State.Tick) >= s.State.NextRewardAt {
		s.offerReward()
	}

	s.checkGameOver()
}

func (s *Simulator) offerReward() {
	// 1. Grant +1 Train automatically every week
	s.State.Resources.Grant(RewardTrain)

	// 2. Offer 2 random bonus choices
	pool := []RewardType{RewardLine, RewardCarriage, RewardTunnel, RewardInterchange}
	rand.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	s.State.PendingRewardChoices = pool[:2]

	// 3. Schedule next reward trigger tick
	s.State.NextRewardAt = float64(s.State.Tick) + rewardInterval()
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

	s.State.Lines = append(s.State.Lines, Line{
		ID:       id,
		Stations: append([]int(nil), a.Stations...),
		Removed:  false,
	})

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

	line.Stations = append(line.Stations, a.StationID)
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
