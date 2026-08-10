package engine

// NetworkGraph is the undirected adjacency representation of the active metro network.
type NetworkGraph struct {
	Adj       map[int][]int    // stationID → neighbour station IDs (deduplicated)
	EdgeLines map[[2]int][]int // canonical (u,v) → line IDs that connect them
}

// BuildGraph constructs a NetworkGraph from active (non-removed) lines and alive stations.
func BuildGraph(state *GameState) NetworkGraph {
	g := NetworkGraph{
		Adj:       make(map[int][]int),
		EdgeLines: make(map[[2]int][]int),
	}
	for _, line := range state.Lines {
		if line.Removed {
			continue
		}
		for i := 0; i+1 < len(line.Stations); i++ {
			u := line.Stations[i]
			v := line.Stations[i+1]
			if u < 0 || u >= len(state.Stations) || !state.Stations[u].Alive {
				continue
			}
			if v < 0 || v >= len(state.Stations) || !state.Stations[v].Alive {
				continue
			}
			g.addEdge(u, v, line.ID)
		}
		// add the wrap-around edge for closed loop lines
		if line.IsLoop && len(line.Stations) >= 2 {
			u := line.Stations[len(line.Stations)-1]
			v := line.Stations[0]
			if u >= 0 && u < len(state.Stations) && state.Stations[u].Alive &&
				v >= 0 && v < len(state.Stations) && state.Stations[v].Alive {
				g.addEdge(u, v, line.ID)
			}
		}
	}
	return g
}

// addEdge adds an undirected edge between u and v attributed to lineID.
func (g *NetworkGraph) addEdge(u, v, lineID int) {
	key := edgeKey(u, v)
	if !contains(g.EdgeLines[key], lineID) {
		g.EdgeLines[key] = append(g.EdgeLines[key], lineID)
	}
	if !contains(g.Adj[u], v) {
		g.Adj[u] = append(g.Adj[u], v)
	}
	if !contains(g.Adj[v], u) {
		g.Adj[v] = append(g.Adj[v], u)
	}
}

// Neighbours returns station IDs directly reachable from stationID via active lines.
func (g *NetworkGraph) Neighbours(stationID int) []int {
	return g.Adj[stationID]
}

// LinesFor returns line IDs that directly connect stations a and b (order-insensitive).
func (g *NetworkGraph) LinesFor(a, b int) []int {
	return g.EdgeLines[edgeKey(a, b)]
}

// edgeKey returns a canonical key for an undirected edge so (u,v) and (v,u) map identically.
func edgeKey(u, v int) [2]int {
	if u <= v {
		return [2]int{u, v}
	}
	return [2]int{v, u}
}

// contains reports whether val is present in slice.
func contains(slice []int, val int) bool {
	for _, x := range slice {
		if x == val {
			return true
		}
	}
	return false
}
