package engine

import "math"

type Observation struct {
	MapName              string
	StationKinds         []StationKind
	StationQueues        []int
	StationCapacities    []int     // max queue per station
	StationTimers        []float64 // overcrowding countdown seconds per station; -1 = not active
	TrainLineIDs         []int
	TrainSegments        []int
	TrainLoads           []int
	Rivers               []RiverSegment
	WaterPolygons        []WaterPolygon
	Resources            ResourcePool
	PendingRewardChoices []RewardType
	AdjacencyList        map[int][]int
	Score                int
	Tick                 uint64
}

func (s *Simulator) Observation() Observation {
	obs := Observation{
		MapName:              s.State.MapName,
		Rivers:               append([]RiverSegment(nil), s.State.Rivers...),
		WaterPolygons:        append([]WaterPolygon(nil), s.State.WaterPolygons...),
		Resources:            s.State.Resources,
		PendingRewardChoices: append([]RewardType(nil), s.State.PendingRewardChoices...),
	}

	for _, st := range s.State.Stations {
		obs.StationKinds = append(obs.StationKinds, st.Kind)
		obs.StationQueues = append(obs.StationQueues, len(st.Queue))
		obs.StationCapacities = append(obs.StationCapacities, st.Capacity)
		obs.StationTimers = append(obs.StationTimers, st.OvercrowdingTimer)
	}

	for _, tr := range s.State.Trains {
		if !tr.Active {
			continue
		}

		obs.TrainLineIDs = append(obs.TrainLineIDs, tr.LineID)
		obs.TrainSegments = append(obs.TrainSegments, tr.Segment)
		obs.TrainLoads = append(obs.TrainLoads, len(tr.Passengers))
	}

	obs.Score = s.State.Score
	obs.Tick = s.State.Tick

	// Copy the adjacency list so the caller cannot mutate the cached graph.
	if len(s.State.Graph.Adj) > 0 {
		obs.AdjacencyList = make(map[int][]int, len(s.State.Graph.Adj))
		for k, v := range s.State.Graph.Adj {
			neighbours := make([]int, len(v))
			copy(neighbours, v)
			obs.AdjacencyList[k] = neighbours
		}
	}

	return obs
}

const (
	NodeFeatureDim   = 32 // PHASE-5: was 29; added incoming_train_count, incoming_train_load, nearest_train_proximity
	EdgeFeatureDim   = 10
	GlobalFeatureDim = 13 // PHASE-2: was 8; added station_count, max_fill, overcrowd_count, pending_reward, game_time
)

type VectorizedObservation struct {
	NumNodes  int
	NumEdges  int
	NodeDim   int
	EdgeDim   int
	GlobalDim int
	Nodes     []float32
	Edges     []int32
	EdgeAttrs []float32
	Globals   []float32
}

func (s *Simulator) WriteVectorizedObservation(outNodes []float32, outEdges []int32, outEdgeAttrs []float32, outGlobals []float32) (numNodes, numEdges int) {
	s.rebuildGraphIfNeeded()

	N := len(s.State.Stations)
	numNodes = N

	// ---- Build lines-per-station lookup (needed for node feature [26]) ----
	linesServingStation := make([]int, N)
	for _, line := range s.State.Lines {
		if line.Removed {
			continue
		}
		seen := make(map[int]bool)
		for _, stID := range line.Stations {
			if stID >= 0 && stID < N && !seen[stID] {
				linesServingStation[stID]++
				seen[stID] = true
			}
		}
	}

	// ---- PHASE-5 Task 21: Build per-station train tracking lookup ----
	type stationTrainInfo struct {
		count        int
		paxCount     int
		totalCap     int
		maxProximity float64
	}
	stTrains := make([]stationTrainInfo, N)

	for trIdx := range s.State.Trains {
		tr := &s.State.Trains[trIdx]
		if !tr.Active || tr.LineID < 0 || tr.LineID >= len(s.State.Lines) {
			continue
		}
		line := &s.State.Lines[tr.LineID]
		if line.Removed || len(line.Stations) < 2 {
			continue
		}

		curStID := -1
		if tr.Segment >= 0 && tr.Segment < len(line.Stations) {
			curStID = line.Stations[tr.Segment]
		}

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
		nextStID := -1
		if nextSegIdx >= 0 && nextSegIdx < len(line.Stations) {
			nextStID = line.Stations[nextSegIdx]
		}

		cap := tr.Capacity
		if cap <= 0 {
			cap = 6
		}
		pax := len(tr.Passengers)
		prog := tr.Progress
		if prog < 0 {
			prog = 0
		}
		if prog > 1 {
			prog = 1
		}

		if curStID >= 0 && curStID < N && (tr.DwellRemaining > 0 || prog < 0.15) {
			info := &stTrains[curStID]
			info.count++
			info.paxCount += pax
			info.totalCap += cap
			if 1.0 > info.maxProximity {
				info.maxProximity = 1.0
			}
		}

		if nextStID >= 0 && nextStID < N && nextStID != curStID {
			info := &stTrains[nextStID]
			info.count++
			info.paxCount += pax
			info.totalCap += cap
			if prog > info.maxProximity {
				info.maxProximity = prog
			}
		}
	}

	for i := 0; i < N; i++ {
		st := &s.State.Stations[i]
		base := i * NodeFeatureDim
		if base+NodeFeatureDim > len(outNodes) {
			break
		}

		outNodes[base+0] = float32(st.Pos.X / 100.0)
		outNodes[base+1] = float32(st.Pos.Y / 100.0)

		for k := 0; k < 10; k++ {
			outNodes[base+2+k] = 0
		}
		if int(st.Kind) >= 0 && int(st.Kind) < 10 {
			outNodes[base+2+int(st.Kind)] = 1.0
		}

		for k := 0; k < 10; k++ {
			outNodes[base+12+k] = 0
		}
		for _, p := range st.Queue {
			if int(p.Destination) >= 0 && int(p.Destination) < 10 {
				outNodes[base+12+int(p.Destination)] += 1.0
			}
		}

		if st.OvercrowdingTimer < 0 {
			outNodes[base+22] = 0.0
		} else {
			outNodes[base+22] = float32(OvercrowdingProgress(st))
		}

		degree := len(s.State.Graph.Neighbours(st.ID))
		outNodes[base+23] = float32(degree)

		if st.IsInterchange {
			outNodes[base+24] = 1.0
		} else {
			outNodes[base+24] = 0.0
		}

		// ---- PHASE-2 new node features ----

		// [25] fill ratio: queue / capacity (0 = empty, 1 = full = game-over imminent)
		cap := st.Capacity
		if cap <= 0 {
			cap = defaultStationCapacity
		}
		outNodes[base+25] = float32(len(st.Queue)) / float32(cap)

		// [26] fraction of lines serving this station (0 = isolated, 1 = all 7 lines)
		outNodes[base+26] = float32(linesServingStation[i]) / 7.0

		// [27] overcrowding timer normalized: 0 if inactive, else (1 - timer/failSeconds)
		// Values near 1.0 = almost dead; 0 = no timer active
		if st.OvercrowdingTimer < 0 {
			outNodes[base+27] = 0.0
		} else {
			outNodes[base+27] = float32(1.0 - st.OvercrowdingTimer/overcrowdingFailureSeconds)
		}

		// [28] total queue normalized by capacity (same signal as [25] but kept separate
		//      so the GNN can distinguish "4/6" from "4/8" even if fill ratio is similar)
		outNodes[base+28] = float32(len(st.Queue)) / float32(defaultStationCapacity)

		// ---- PHASE-5 new node features (Task 21) ----

		// [29] incoming/present trains count normalized (1 train = 0.25, 4+ trains = 1.0)
		tInfo := stTrains[i]
		outNodes[base+29] = float32(tInfo.count) / 4.0
		if outNodes[base+29] > 1.0 {
			outNodes[base+29] = 1.0
		}

		// [30] incoming/present train passenger load (0.0 = empty or no trains, 1.0 = fully packed)
		if tInfo.totalCap > 0 {
			outNodes[base+30] = float32(tInfo.paxCount) / float32(tInfo.totalCap)
			if outNodes[base+30] > 1.0 {
				outNodes[base+30] = 1.0
			}
		} else {
			outNodes[base+30] = 0.0
		}

		// [31] nearest train arrival proximity in [0, 1] (1.0 = at platform / arriving now, 0.0 = no train)
		outNodes[base+31] = float32(tInfo.maxProximity)
	}

	edgeCount := 0
	for _, line := range s.State.Lines {
		if line.Removed || len(line.Stations) < 2 {
			continue
		}
		for i := 0; i+1 < len(line.Stations); i++ {
			u := line.Stations[i]
			v := line.Stations[i+1]
			if u < 0 || u >= N || v < 0 || v >= N {
				continue
			}

			if edgeCount*2+1 < len(outEdges) {
				outEdges[edgeCount*2] = int32(u)
				outEdges[edgeCount*2+1] = int32(v)
			}

			baseAttr := edgeCount * EdgeFeatureDim
			if baseAttr+EdgeFeatureDim <= len(outEdgeAttrs) {
				for k := 0; k < 7; k++ {
					outEdgeAttrs[baseAttr+k] = 0
				}
				if line.ID >= 0 && line.ID < 7 {
					outEdgeAttrs[baseAttr+line.ID] = 1.0
				}
				uPos := s.State.Stations[u].Pos
				vPos := s.State.Stations[v].Pos
				outEdgeAttrs[baseAttr+7] = float32(distance(uPos, vPos) / 100.0)
				outEdgeAttrs[baseAttr+8] = 1.0
				if i < len(line.TunnelAt) && line.TunnelAt[i] {
					outEdgeAttrs[baseAttr+9] = 1.0
				} else {
					outEdgeAttrs[baseAttr+9] = 0.0
				}
			}
			edgeCount++

			if edgeCount*2+1 < len(outEdges) {
				outEdges[edgeCount*2] = int32(v)
				outEdges[edgeCount*2+1] = int32(u)
			}

			baseAttr = edgeCount * EdgeFeatureDim
			if baseAttr+EdgeFeatureDim <= len(outEdgeAttrs) {
				for k := 0; k < 7; k++ {
					outEdgeAttrs[baseAttr+k] = 0
				}
				if line.ID >= 0 && line.ID < 7 {
					outEdgeAttrs[baseAttr+line.ID] = 1.0
				}
				uPos := s.State.Stations[u].Pos
				vPos := s.State.Stations[v].Pos
				outEdgeAttrs[baseAttr+7] = float32(distance(uPos, vPos) / 100.0)
				outEdgeAttrs[baseAttr+8] = -1.0
				if i < len(line.TunnelAt) && line.TunnelAt[i] {
					outEdgeAttrs[baseAttr+9] = 1.0
				} else {
					outEdgeAttrs[baseAttr+9] = 0.0
				}
			}
			edgeCount++
		}
	}
	numEdges = edgeCount

	if len(outGlobals) >= GlobalFeatureDim {
		outGlobals[0] = float32(s.State.Resources.Lines)
		outGlobals[1] = float32(s.State.Resources.Trains)
		outGlobals[2] = float32(s.State.Resources.Carriages)
		outGlobals[3] = float32(s.State.Resources.Tunnels)
		outGlobals[4] = float32(s.State.Resources.Interchanges)
		weekSeconds := float64(rewardInterval()) / 30.0
		outGlobals[5] = float32(math.Mod(s.State.GameTimeSeconds, weekSeconds) / weekSeconds)
		// PHASE-1 fix BUG-D: normalize score so it stays in [0,~1] range like all other
		// global features. Raw cumulative score (0–10000+) dominated global_proj MLP gradients.
		outGlobals[6] = float32(s.State.Score) / 500.0
		activeTrains := 0
		for _, tr := range s.State.Trains {
			if tr.Active {
				activeTrains++
			}
		}
		outGlobals[7] = float32(activeTrains)

		// ---- PHASE-2 new global features ----

		// [8] station count normalized (how large is the current network)
		outGlobals[8] = float32(N) / 30.0

		// [9] max fill ratio across all stations (best single predictor of imminent game-over)
		maxFill := float32(0)
		numOvercrowding := 0
		for i := 0; i < N; i++ {
			st := &s.State.Stations[i]
			cap := st.Capacity
			if cap <= 0 {
				cap = defaultStationCapacity
			}
			fill := float32(len(st.Queue)) / float32(cap)
			if fill > maxFill {
				maxFill = fill
			}
			if st.OvercrowdingTimer >= 0 {
				numOvercrowding++
			}
		}
		outGlobals[9] = maxFill

		// [10] fraction of stations currently overcrowding
		if N > 0 {
			outGlobals[10] = float32(numOvercrowding) / float32(N)
		} else {
			outGlobals[10] = 0
		}

		// [11] pending reward choices flag (1 = agent must choose a reward, all other actions masked)
		if len(s.State.PendingRewardChoices) > 0 {
			outGlobals[11] = 1.0
		} else {
			outGlobals[11] = 0.0
		}

		// [12] game time fraction (capped at 1 for very long games ~1 hour)
		outGlobals[12] = float32(math.Min(s.State.GameTimeSeconds/3600.0, 1.0))
	}

	return numNodes, numEdges
}

func (s *Simulator) VectorizedObservation() VectorizedObservation {
	N := len(s.State.Stations)
	nodes := make([]float32, N*NodeFeatureDim)
	edges := make([]int32, 600*2)
	edgeAttrs := make([]float32, 600*EdgeFeatureDim)
	globals := make([]float32, GlobalFeatureDim)

	numNodes, numEdges := s.WriteVectorizedObservation(nodes, edges, edgeAttrs, globals)

	return VectorizedObservation{
		NumNodes:  numNodes,
		NumEdges:  numEdges,
		NodeDim:   NodeFeatureDim,
		EdgeDim:   EdgeFeatureDim,
		GlobalDim: GlobalFeatureDim,
		Nodes:     nodes[:numNodes*NodeFeatureDim],
		Edges:     edges[:numEdges*2],
		EdgeAttrs: edgeAttrs[:numEdges*EdgeFeatureDim],
		Globals:   globals,
	}
}
