package engine

import (
	"errors"
)

const (
	MaxStations = 30
	MaxLines    = 7
	MaxTrains   = 28
)

// ActionSpace layout ranges
const (
	ActionNoOp = 0
	
	// AddLine: station u < v (30 * 29 / 2 = 435 combinations)
	AddLineOffset = 1
	AddLineCount  = (MaxStations * (MaxStations - 1)) / 2

	// ExtendLine: line_id (0..6) * station_id (0..29) * end (0=front, 1=back) = 7 * 30 * 2 = 420
	ExtendLineOffset = AddLineOffset + AddLineCount
	ExtendLineCount  = MaxLines * MaxStations * 2

	// InsertStation: line_id (0..6) * station_id (0..29) * segment_idx (1..15) = 7 * 30 * 15 = 3150
	InsertStationOffset = ExtendLineOffset + ExtendLineCount
	InsertStationCount  = MaxLines * MaxStations * 15

	// AddTrain: line_id (0..6) = 7
	AddTrainOffset = InsertStationOffset + InsertStationCount
	AddTrainCount  = MaxLines

	// AddCarriage: train_id (0..27) = 28
	AddCarriageOffset = AddTrainOffset + AddTrainCount
	AddCarriageCount  = MaxTrains

	// UpgradeInterchange: station_id (0..29) = 30
	UpgradeInterchangeOffset = AddCarriageOffset + AddCarriageCount
	UpgradeInterchangeCount  = MaxStations

	// ChooseReward: choice (0..3) = 4
	ChooseRewardOffset = UpgradeInterchangeOffset + UpgradeInterchangeCount
	ChooseRewardCount  = 4

	// CloseLoop: line_id (0..6) = 7
	CloseLoopOffset = ChooseRewardOffset + ChooseRewardCount
	CloseLoopCount  = MaxLines

	// OpenLoop: line_id (0..6) = 7
	OpenLoopOffset = CloseLoopOffset + CloseLoopCount
	OpenLoopCount  = MaxLines

	// RemoveLine: line_id (0..6) = 7
	RemoveLineOffset = OpenLoopOffset + OpenLoopCount
	RemoveLineCount  = MaxLines

	// ShortenLine: line_id (0..6) * from_front (0..1) = 14
	ShortenLineOffset = RemoveLineOffset + RemoveLineCount
	ShortenLineCount  = MaxLines * 2

	TotalActionSpaceSize = ShortenLineOffset + ShortenLineCount
)

func MaxActionSpaceSize() int {
	return TotalActionSpaceSize
}

// ActionFromIndex translates a flat integer action ID into a typed Action struct.
func ActionFromIndex(id int) (Action, error) {
	if id == ActionNoOp {
		return nil, nil
	}

	if id >= AddLineOffset && id < AddLineOffset+AddLineCount {
		idx := id - AddLineOffset
		u, v := 0, 0
		curr := 0
		found := false
		for i := 0; i < MaxStations; i++ {
			for j := i + 1; j < MaxStations; j++ {
				if curr == idx {
					u, v = i, j
					found = true
					break
				}
				curr++
			}
			if found {
				break
			}
		}
		return AddLine{Stations: []int{u, v}}, nil
	}

	if id >= ExtendLineOffset && id < ExtendLineOffset+ExtendLineCount {
		idx := id - ExtendLineOffset
		fromFront := (idx % 2) == 0
		rem := idx / 2
		stID := rem % MaxStations
		lineID := rem / MaxStations
		return ExtendLine{LineID: lineID, StationID: stID, FromFront: fromFront}, nil
	}

	if id >= InsertStationOffset && id < InsertStationOffset+InsertStationCount {
		idx := id - InsertStationOffset
		segIdx := (idx % 15) + 1
		rem := idx / 15
		stID := rem % MaxStations
		lineID := rem / MaxStations
		return InsertStation{LineID: lineID, StationID: stID, Index: segIdx}, nil
	}

	if id >= AddTrainOffset && id < AddTrainOffset+AddTrainCount {
		lineID := id - AddTrainOffset
		return AddTrain{LineID: lineID}, nil
	}

	if id >= AddCarriageOffset && id < AddCarriageOffset+AddCarriageCount {
		trID := id - AddCarriageOffset
		return AddCarriage{TrainID: trID}, nil
	}

	if id >= UpgradeInterchangeOffset && id < UpgradeInterchangeOffset+UpgradeInterchangeCount {
		stID := id - UpgradeInterchangeOffset
		return UpgradeInterchange{StationID: stID}, nil
	}

	if id >= ChooseRewardOffset && id < ChooseRewardOffset+ChooseRewardCount {
		choiceIdx := id - ChooseRewardOffset
		return ChooseReward{Choice: RewardType(choiceIdx)}, nil
	}

	if id >= CloseLoopOffset && id < CloseLoopOffset+CloseLoopCount {
		lineID := id - CloseLoopOffset
		return CloseLoop{LineID: lineID}, nil
	}

	if id >= OpenLoopOffset && id < OpenLoopOffset+OpenLoopCount {
		lineID := id - OpenLoopOffset
		return OpenLoop{LineID: lineID}, nil
	}

	if id >= RemoveLineOffset && id < RemoveLineOffset+RemoveLineCount {
		lineID := id - RemoveLineOffset
		return RemoveLine{LineID: lineID}, nil
	}

	if id >= ShortenLineOffset && id < ShortenLineOffset+ShortenLineCount {
		idx := id - ShortenLineOffset
		fromFront := (idx % 2) == 0
		lineID := idx / 2
		return ShortenLine{LineID: lineID, FromFront: fromFront}, nil
	}

	return nil, errors.New("invalid action index")
}

// GetActionMask evaluates current state constraints and returns a boolean slice of length TotalActionSpaceSize
// indicating which actions are legally executable at the current tick.
func (s *Simulator) GetActionMask(outMask []bool) []bool {
	if len(outMask) < TotalActionSpaceSize {
		outMask = make([]bool, TotalActionSpaceSize)
	} else {
		for i := 0; i < TotalActionSpaceSize; i++ {
			outMask[i] = false
		}
	}

	if !s.State.Alive {
		return outMask
	}

	// Action 0: NoOp is always valid
	outMask[ActionNoOp] = true

	// If pending reward choices exist, only ChooseReward actions are valid
	if len(s.State.PendingRewardChoices) > 0 {
		for cIdx := 0; cIdx < ChooseRewardCount; cIdx++ {
			if cIdx < len(s.State.PendingRewardChoices) {
				outMask[ChooseRewardOffset+cIdx] = true
			}
		}
		return outMask
	}

	N := len(s.State.Stations)

	// 1. AddLine
	if s.State.Resources.CanSpend(RewardLine) {
		currIdx := 0
		for u := 0; u < MaxStations; u++ {
			for v := u + 1; v < MaxStations; v++ {
				if u < N && v < N && s.State.Stations[u].Alive && s.State.Stations[v].Alive {
					uPos := s.State.Stations[u].Pos
					vPos := s.State.Stations[v].Pos
					needsTunnel := CrossesWater(uPos, vPos, s.State.Rivers, s.State.WaterPolygons)
					if !needsTunnel || s.State.Resources.CanSpend(RewardTunnel) {
						outMask[AddLineOffset+currIdx] = true
					}
				}
				currIdx++
			}
		}
	}

	// 2. ExtendLine
	for lID := 0; lID < len(s.State.Lines); lID++ {
		line := &s.State.Lines[lID]
		if line.Removed || len(line.Stations) == 0 {
			continue
		}

		for stID := 0; stID < N; stID++ {
			if !s.State.Stations[stID].Alive {
				continue
			}

			// check if station is already on line
			already := false
			for _, sIdx := range line.Stations {
				if sIdx == stID {
					already = true
					break
				}
			}
			if already {
				continue
			}

			// Front extension
			endFront := line.Stations[0]
			needsTunnelF := CrossesWater(s.State.Stations[endFront].Pos, s.State.Stations[stID].Pos, s.State.Rivers, s.State.WaterPolygons)
			if !needsTunnelF || s.State.Resources.CanSpend(RewardTunnel) {
				idx := (lID*MaxStations+stID)*2 + 0
				if idx < ExtendLineCount {
					outMask[ExtendLineOffset+idx] = true
				}
			}

			// Back extension
			endBack := line.Stations[len(line.Stations)-1]
			needsTunnelB := CrossesWater(s.State.Stations[endBack].Pos, s.State.Stations[stID].Pos, s.State.Rivers, s.State.WaterPolygons)
			if !needsTunnelB || s.State.Resources.CanSpend(RewardTunnel) {
				idx := (lID*MaxStations+stID)*2 + 1
				if idx < ExtendLineCount {
					outMask[ExtendLineOffset+idx] = true
				}
			}
		}
	}

	// 3. AddTrain
	if s.State.Resources.CanSpend(RewardTrain) {
		for lID := 0; lID < len(s.State.Lines); lID++ {
			line := &s.State.Lines[lID]
			if line.Removed || len(line.Stations) < 2 {
				continue
			}
			activeCount := 0
			for _, tr := range s.State.Trains {
				if tr.Active && tr.LineID == lID {
					activeCount++
				}
			}
			maxLimit := s.State.MaxTrainsPerLine
			if maxLimit <= 0 {
				maxLimit = 4
			}
			if activeCount < maxLimit {
				outMask[AddTrainOffset+lID] = true
			}
		}
	}

	// 4. AddCarriage
	if s.State.Resources.CanSpend(RewardCarriage) {
		for trID := 0; trID < len(s.State.Trains); trID++ {
			if s.State.Trains[trID].Active && trID < AddCarriageCount {
				outMask[AddCarriageOffset+trID] = true
			}
		}
	}

	// 5. UpgradeInterchange
	if s.State.Resources.CanSpend(RewardInterchange) {
		for stID := 0; stID < N; stID++ {
			st := &s.State.Stations[stID]
			if st.Alive && !st.IsInterchange && stID < UpgradeInterchangeCount {
				outMask[UpgradeInterchangeOffset+stID] = true
			}
		}
	}

	// 6. CloseLoop / OpenLoop / RemoveLine / ShortenLine
	for lID := 0; lID < len(s.State.Lines); lID++ {
		line := &s.State.Lines[lID]
		if line.Removed {
			continue
		}

		if !line.IsLoop && len(line.Stations) >= 2 {
			firstPos := s.State.Stations[line.Stations[0]].Pos
			lastPos := s.State.Stations[line.Stations[len(line.Stations)-1]].Pos
			needsTunnel := CrossesWater(lastPos, firstPos, s.State.Rivers, s.State.WaterPolygons)
			if !needsTunnel || s.State.Resources.CanSpend(RewardTunnel) {
				outMask[CloseLoopOffset+lID] = true
			}
		}

		if line.IsLoop {
			outMask[OpenLoopOffset+lID] = true
		}

		outMask[RemoveLineOffset+lID] = true

		if len(line.Stations) > 2 && !line.IsLoop {
			outMask[ShortenLineOffset+lID*2+0] = true
			outMask[ShortenLineOffset+lID*2+1] = true
		}
	}

	return outMask
}
