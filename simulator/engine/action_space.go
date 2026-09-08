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

// ActionToIndex translates a typed Action struct into a flat integer action ID.
func ActionToIndex(action Action) (int, bool) {
	if action == nil {
		return ActionNoOp, true
	}
	switch a := action.(type) {
	case AddLine:
		if len(a.Stations) >= 2 {
			u, v := a.Stations[0], a.Stations[1]
			if u > v {
				u, v = v, u
			}
			curr := 0
			for i := 0; i < MaxStations; i++ {
				for j := i + 1; j < MaxStations; j++ {
					if i == u && j == v {
						return AddLineOffset + curr, true
					}
					curr++
				}
			}
		}
	case ExtendLine:
		fromFront := 0
		if !a.FromFront {
			fromFront = 1
		}
		idx := (a.LineID*MaxStations+a.StationID)*2 + fromFront
		if idx >= 0 && idx < ExtendLineCount {
			return ExtendLineOffset + idx, true
		}
	case InsertStation:
		idx := (a.LineID*MaxStations+a.StationID)*15 + a.Index
		if idx >= 0 && idx < InsertStationCount {
			return InsertStationOffset + idx, true
		}
	case AddTrain:
		if a.LineID >= 0 && a.LineID < AddTrainCount {
			return AddTrainOffset + a.LineID, true
		}
	case AddCarriage:
		if a.LineID >= 0 && a.LineID < AddCarriageCount {
			return AddCarriageOffset + a.LineID, true
		}
	case UpgradeInterchange:
		if a.StationID >= 0 && a.StationID < UpgradeInterchangeCount {
			return UpgradeInterchangeOffset + a.StationID, true
		}
	case CloseLoop:
		if a.LineID >= 0 && a.LineID < CloseLoopCount {
			return CloseLoopOffset + a.LineID, true
		}
	case OpenLoop:
		if a.LineID >= 0 && a.LineID < OpenLoopCount {
			return OpenLoopOffset + a.LineID, true
		}
	case ChooseReward:
		idx := int(a.Choice)
		if idx >= 0 && idx < ChooseRewardCount {
			return ChooseRewardOffset + idx, true
		}
	}
	return -1, false
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

	// Action 0: NoOp is valid only if no reward choice is pending
	outMask[ActionNoOp] = len(s.State.PendingRewardChoices) == 0

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
	if s.State.Resources.CanSpend(RewardLine) && s.State.Resources.CanSpend(RewardTrain) {
		s.rebuildGraphIfNeeded()

		// Count alive stations with degree 0 (isolated stations)
		isolatedCount := 0
		for stID := 0; stID < N; stID++ {
			if s.State.Stations[stID].Alive && s.stationDegree(stID) == 0 {
				isolatedCount++
			}
		}

		currIdx := 0
		for u := 0; u < MaxStations; u++ {
			for v := u + 1; v < MaxStations; v++ {
				if u < N && v < N && s.State.Stations[u].Alive && s.State.Stations[v].Alive {
					// 1) Never create duplicate direct parallel track between the same station pair
					if !s.hasDirectSegment(u, v) {
						degU := s.stationDegree(u)
						degV := s.stationDegree(v)

						// 2) Anti-redundancy constraint:
						// If any stations are isolated, require the new line to connect at least one isolated station
						// OR bridge two disconnected components.
						// If all stations are already served, only allow lines that bridge disconnected components.
						canReach := s.CanReach(u, v)
						allow := false
						if isolatedCount > 0 {
							if degU == 0 || degV == 0 || !canReach {
								allow = true
							}
						} else {
							if !canReach {
								allow = true
							}
						}

						if allow {
							uPos := s.State.Stations[u].Pos
							vPos := s.State.Stations[v].Pos
							needsTunnel := CrossesWater(uPos, vPos, s.State.Rivers, s.State.WaterPolygons)
							if !needsTunnel || s.State.Resources.CanSpend(RewardTunnel) {
								outMask[AddLineOffset+currIdx] = true
							}
						}
					}
				}
				currIdx++
			}
		}
	}

	// 2. ExtendLine
	for lID := 0; lID < len(s.State.Lines); lID++ {
		line := &s.State.Lines[lID]
		if line.Removed || line.IsLoop || len(line.Stations) == 0 {
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
			if !s.hasDirectSegment(endFront, stID) {
				needsTunnelF := CrossesWater(s.State.Stations[endFront].Pos, s.State.Stations[stID].Pos, s.State.Rivers, s.State.WaterPolygons)
				if !needsTunnelF || s.State.Resources.CanSpend(RewardTunnel) {
					idx := (lID*MaxStations+stID)*2 + 0
					if idx < ExtendLineCount {
						outMask[ExtendLineOffset+idx] = true
					}
				}
			}

			// Back extension
			endBack := line.Stations[len(line.Stations)-1]
			if !s.hasDirectSegment(endBack, stID) {
				needsTunnelB := CrossesWater(s.State.Stations[endBack].Pos, s.State.Stations[stID].Pos, s.State.Rivers, s.State.WaterPolygons)
				if !needsTunnelB || s.State.Resources.CanSpend(RewardTunnel) {
					idx := (lID*MaxStations+stID)*2 + 1
					if idx < ExtendLineCount {
						outMask[ExtendLineOffset+idx] = true
					}
				}
			}
		}
	}

	// 2b. InsertStation
	for lID := 0; lID < len(s.State.Lines); lID++ {
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
				prevSt := line.Stations[segIdx-1]
				nextSt := line.Stations[segIdx]
				if s.hasDirectSegment(prevSt, stID) || s.hasDirectSegment(stID, nextSt) {
					continue
				}
				stPrevPos := s.State.Stations[prevSt].Pos
				stNextPos := s.State.Stations[nextSt].Pos

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
	for lID := 0; lID < len(s.State.Lines) && lID < MaxLines; lID++ {
		line := &s.State.Lines[lID]
		if line.Removed {
			continue
		}

		// Cooldown: prevent toggling loop state within 10 seconds of last toggle
		if s.loopToggled[lID] && s.State.GameTimeSeconds-s.lastLoopToggleTime[lID] < 10.0 {
			continue
		}

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

		// Action disabled per user request to prevent AI from messing up passenger progress:
		// outMask[RemoveLineOffset+lID] = true
		// 
		// if len(line.Stations) > 2 && !line.IsLoop {
		// 	outMask[ShortenLineOffset+lID*2+0] = true
		// 	outMask[ShortenLineOffset+lID*2+1] = true
		// }
	}

	return outMask
}
