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

	// PHASE-4: AddCarriage now indexed by lineID (0..6), not trainID (0..27).
	// Agent can meaningfully correlate "add carriage to line X" with the line it knows.
	// Action space: 7 (was 28).
	AddCarriageOffset = AddTrainOffset + AddTrainCount
	AddCarriageCount  = MaxLines

	// UpgradeInterchange: station_id (0..29) = 30
	UpgradeInterchangeOffset = AddCarriageOffset + AddCarriageCount
	UpgradeInterchangeCount  = MaxStations

	// ChooseReward: positional index into 2 offered choices (0 or 1)
	ChooseRewardOffset = UpgradeInterchangeOffset + UpgradeInterchangeCount
	ChooseRewardCount  = 2

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
		// PHASE-4: decode lineID (not trainID)
		lineID := id - AddCarriageOffset
		return AddCarriage{LineID: lineID}, nil
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

	// If pending reward choices exist, only ChooseReward actions are valid
	if len(s.State.PendingRewardChoices) > 0 {
		for cIdx := 0; cIdx < ChooseRewardCount; cIdx++ {
			if cIdx < len(s.State.PendingRewardChoices) {
				outMask[ChooseRewardOffset+cIdx] = true
			}
		}
		return outMask
	}

	// Action 0: NoOp is always valid
	outMask[ActionNoOp] = true

	N := len(s.State.Stations)

	// 1. AddLine
	if s.State.Resources.CanSpend(RewardLine) && s.State.Resources.CanSpend(RewardTrain) {
		currIdx := 0
		for u := 0; u < MaxStations; u++ {
			for v := u + 1; v < MaxStations; v++ {
				if u < N && v < N && s.State.Stations[u].Alive && s.State.Stations[v].Alive {
					// Check if any active line already directly connects station u and station v
					alreadyDirect := false
					for _, line := range s.State.Lines {
						if line.Removed || len(line.Stations) < 2 {
							continue
						}
						stList := line.Stations
						for i := 0; i+1 < len(stList); i++ {
							if (stList[i] == u && stList[i+1] == v) || (stList[i] == v && stList[i+1] == u) {
								alreadyDirect = true
								break
							}
						}
						if !alreadyDirect && line.IsLoop && len(stList) >= 3 {
							if (stList[0] == u && stList[len(stList)-1] == v) || (stList[0] == v && stList[len(stList)-1] == u) {
								alreadyDirect = true
								break
							}
						}
						if alreadyDirect {
							break
						}
					}

					uPos := s.State.Stations[u].Pos
					vPos := s.State.Stations[v].Pos
					needsTunnel := CrossesWater(uPos, vPos, s.State.Rivers, s.State.WaterPolygons)
					if !alreadyDirect && (!needsTunnel || s.State.Resources.CanSpend(RewardTunnel)) {
						outMask[AddLineOffset+currIdx] = true
					}
				}
				currIdx++
			}
		}
	}

	// 2. ExtendLine
	for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
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

	// 2b. InsertStation
	for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
		line := &s.State.Lines[lID]
		if line.Removed || len(line.Stations) < 2 {
			continue
		}
		n := len(line.Stations)
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

			stNewPos := s.State.Stations[stID].Pos
			for segIdx := 1; segIdx < n && segIdx <= 15; segIdx++ {
				stPrevPos := s.State.Stations[line.Stations[segIdx-1]].Pos
				stNextPos := s.State.Stations[line.Stations[segIdx]].Pos

				cross1 := CrossesWater(stPrevPos, stNewPos, s.State.Rivers, s.State.WaterPolygons)
				cross2 := CrossesWater(stNewPos, stNextPos, s.State.Rivers, s.State.WaterPolygons)

				origTunnel := false
				if segIdx-1 < len(line.TunnelAt) && line.TunnelAt[segIdx-1] {
					origTunnel = true
				}

				netTunnels := 0
				if cross1 {
					netTunnels++
				}
				if cross2 {
					netTunnels++
				}
				if origTunnel {
					netTunnels--
				}

				if netTunnels <= 0 || s.State.Resources.CanSpend(RewardTunnel) {
					idx := (lID*MaxStations+stID)*15 + (segIdx - 1)
					if idx < InsertStationCount {
						outMask[InsertStationOffset+idx] = true
					}
				}
			}
		}
	}

	// 3. AddTrain
	if s.State.Resources.CanSpend(RewardTrain) {
		for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
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

	// 4. AddCarriage — PHASE-4: index by lineID so agent can target underserved lines.
	if s.State.Resources.CanSpend(RewardCarriage) {
		for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
			line := &s.State.Lines[lID]
			if line.Removed {
				continue
			}
			// Enable if at least one active train is on this line
			for _, tr := range s.State.Trains {
				if tr.Active && tr.LineID == lID {
					outMask[AddCarriageOffset+lID] = true
					break
				}
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
	const LoopToggleCooldownTicks = 900 // P1-4: 30 seconds at 30 Hz

	for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
		line := &s.State.Lines[lID]
		if line.Removed {
			continue
		}

		// P1-4: Loop Toggling Hysteresis & Cooldown
		// Mask CloseLoop and OpenLoop for 900 ticks (30 seconds) following any loop toggle.
		canToggleLoop := !line.HasBeenLoopToggled || (s.State.Tick >= line.LastLoopToggleTick+LoopToggleCooldownTicks)
		if canToggleLoop {
			// BUG-10 fix: closeLoop() in simulator.go rejects lines with fewer than 3 stations.
			// The mask previously enabled CloseLoop for >= 2 stations, which would always
			// produce an engine error — breaking the mask contract for RL agents / API callers.
			if !line.IsLoop && len(line.Stations) >= 3 {
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
		}

		// P4-1: Safe Dynamic Line Re-Routing & Deletion
		if !line.Removed && len(line.Stations) >= 2 {
			outMask[RemoveLineOffset+lID] = true
		}

		if !line.Removed && len(line.Stations) > 2 && !line.IsLoop {
			outMask[ShortenLineOffset+lID*2+0] = true
			outMask[ShortenLineOffset+lID*2+1] = true
		}
	}

	return outMask
}

