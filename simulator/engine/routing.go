package engine

// RouteInfo is the result of a passenger route query.
type RouteInfo struct {
	Reachable bool // true if a path exists through the network to any destKind station
	IsDirect  bool // true if origin and a destKind station share one active line
}

// CanReach returns true if fromID can reach any alive destKind station via the graph.
func CanReach(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) bool {
	return FindRoute(g, state, fromID, destKind).Reachable
}

// FindRoute checks for a direct (same-line) or transfer (BFS) path from fromID to any destKind station.
func FindRoute(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) RouteInfo {
	if fromID < 0 || fromID >= len(state.Stations) {
		return RouteInfo{}
	}

	if state.Stations[fromID].Kind == destKind {
		return RouteInfo{Reachable: true, IsDirect: true}
	}

	direct := isDirectRoute(state, fromID, destKind)
	reachable := direct || bfsReachable(g, state, fromID, destKind)

	return RouteInfo{Reachable: reachable, IsDirect: direct}
}

// isDirectRoute returns true if fromID and any alive destKind station share at least one active line.
func isDirectRoute(state *GameState, fromID int, destKind StationKind) bool {
	for _, line := range state.Lines {
		if line.Removed {
			continue
		}
		hasOrigin := false
		hasDest := false
		for _, sid := range line.Stations {
			if sid < 0 || sid >= len(state.Stations) {
				continue
			}
			st := &state.Stations[sid]
			if !st.Alive {
				continue
			}
			if sid == fromID {
				hasOrigin = true
			}
			if st.Kind == destKind && sid != fromID {
				hasDest = true
			}
			if hasOrigin && hasDest {
				return true
			}
		}
	}
	return false
}

// bfsReachable returns true if any station reachable from fromID has Kind == destKind.
func bfsReachable(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) bool {
	if len(g.Adj) == 0 {
		return false
	}
	visited := make(map[int]bool)
	queue := []int{fromID}
	visited[fromID] = true
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range g.Neighbours(cur) {
			if visited[nb] {
				continue
			}
			visited[nb] = true
			if nb < 0 || nb >= len(state.Stations) {
				continue
			}
			st := &state.Stations[nb]
			if !st.Alive {
				continue
			}
			if st.Kind == destKind {
				return true
			}
			queue = append(queue, nb)
		}
	}
	return false
}
