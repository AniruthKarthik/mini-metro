package engine

type Observation struct {
	MapName              string
	StationKinds         []StationKind
	StationQueues        []int
	StationCapacities    []int     // max queue per station
	StationTimers        []float64 // overcrowding countdown per station; -1 = not active
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
	NodeFeatureDim   = 25
	EdgeFeatureDim   = 10
	GlobalFeatureDim = 8
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
			outNodes[base+22] = float32(1.0 - st.OvercrowdingTimer/overcrowdingGrace)
			if outNodes[base+22] < 0 {
				outNodes[base+22] = 0.0
			}
		}

		degree := len(s.State.Graph.Neighbours(st.ID))
		outNodes[base+23] = float32(degree)

		if st.IsInterchange {
			outNodes[base+24] = 1.0
		} else {
			outNodes[base+24] = 0.0
		}
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
		outGlobals[5] = float32(s.State.Tick%rewardInterval()) / float32(rewardInterval())
		outGlobals[6] = float32(s.State.Score)
		activeTrains := 0
		for _, tr := range s.State.Trains {
			if tr.Active {
				activeTrains++
			}
		}
		outGlobals[7] = float32(activeTrains)
	}

	return numNodes, numEdges
}

func (s *Simulator) VectorizedObservation() VectorizedObservation {
	N := len(s.State.Stations)
	nodes := make([]float32, N*NodeFeatureDim)
	edges := make([]int32, 200*2)
	edgeAttrs := make([]float32, 200*EdgeFeatureDim)
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