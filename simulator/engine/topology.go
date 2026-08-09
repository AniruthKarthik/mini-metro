package engine

// NetworkGraph is an undirected adjacency representation of the active metro network.
//
// Adj maps each stationID to the deduplicated list of neighbour station IDs that
// are reachable by at least one active (non-removed) line.
//
// EdgeLines maps a canonical (lo, hi) station-ID pair to the list of line IDs
// that directly connect those two stations. This is used by route-finding logic
// (todo items 5 & 6) to detect which lines serve a given edge and to identify
// interchange stations where multiple lines meet.
type NetworkGraph struct {
	Adj       map[int][]int   // stationID → neighbour station IDs (deduplicated)
	EdgeLines map[[2]int][]int // canonical (u,v) pair → line IDs
}

// BuildGraph constructs a fresh NetworkGraph from the current GameState.
// Only non-removed lines and alive stations are included.
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

			// Skip edges where either station is not alive.
			if u < 0 || u >= len(state.Stations) || !state.Stations[u].Alive {
				continue
			}
			if v < 0 || v >= len(state.Stations) || !state.Stations[v].Alive {
				continue
			}

			g.addEdge(u, v, line.ID)
		}
	}

	return g
}

// addEdge adds an undirected edge between u and v attributed to lineID.
// Duplicate neighbour entries are avoided by checking before appending.
func (g *NetworkGraph) addEdge(u, v, lineID int) {
	key := edgeKey(u, v)
	g.EdgeLines[key] = append(g.EdgeLines[key], lineID)

	if !contains(g.Adj[u], v) {
		g.Adj[u] = append(g.Adj[u], v)
	}
	if !contains(g.Adj[v], u) {
		g.Adj[v] = append(g.Adj[v], u)
	}
}

// Neighbours returns the slice of station IDs directly reachable from stationID
// via at least one active line. Returns nil if the station has no connections.
func (g *NetworkGraph) Neighbours(stationID int) []int {
	return g.Adj[stationID]
}

// LinesFor returns the IDs of all lines that directly connect stations a and b.
// Order of a and b does not matter. Returns nil if no direct connection exists.
func (g *NetworkGraph) LinesFor(a, b int) []int {
	return g.EdgeLines[edgeKey(a, b)]
}

// edgeKey returns a canonical [2]int key for an undirected edge so that
// (u,v) and (v,u) map to the same entry.
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
