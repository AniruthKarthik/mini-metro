package engine

import "container/heap"

const transferPenalty = 5.0 // extra Dijkstra cost per line change, discourages unnecessary transfers

// RouteInfo is the result of an A* route query from a station to a destination kind.
type RouteInfo struct {
	Reachable  bool    // true if a path exists to any alive destKind station
	IsDirect   bool    // true if reachable with zero line changes
	Transfers  int     // number of line changes in the optimal path
	NextLineID int     // first line to board; -1 if already at dest or unreachable
	TotalCost  float64 // Dijkstra cost of the optimal path
}

// routeNode is a Dijkstra priority-queue element indexed on (stationID, lineID).
type routeNode struct {
	stationID int
	lineID    int // -1 = not yet on any line (origin state)
	cost      float64
	index     int
}

type routeHeap []*routeNode

func (h routeHeap) Len() int            { return len(h) }
func (h routeHeap) Less(i, j int) bool  { return h[i].cost < h[j].cost }
func (h routeHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i]; h[i].index = i; h[j].index = j }
func (h *routeHeap) Push(x interface{}) { n := x.(*routeNode); n.index = len(*h); *h = append(*h, n) }
func (h *routeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

type stateKey struct{ stationID, lineID int }

// FindOptimalRoute runs Dijkstra on the (stationID, lineID) state space.
// Each hop on the same line costs 1; changing lines costs 1 + transferPenalty.
// The first goal node popped is globally optimal (Dijkstra guarantee on non-negative weights).
func FindOptimalRoute(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) RouteInfo {
	if fromID < 0 || fromID >= len(state.Stations) {
		return RouteInfo{NextLineID: -1}
	}
	if state.Stations[fromID].Kind == destKind {
		return RouteInfo{Reachable: true, IsDirect: true, NextLineID: -1}
	}
	if len(g.Adj) == 0 {
		return RouteInfo{NextLineID: -1}
	}

	dist := make(map[stateKey]float64)
	prev := make(map[stateKey]stateKey)

	h := &routeHeap{}
	heap.Init(h)

	start := stateKey{fromID, -1}
	dist[start] = 0
	heap.Push(h, &routeNode{stationID: fromID, lineID: -1, cost: 0})

	var goalKey stateKey
	found := false

	for h.Len() > 0 {
		cur := heap.Pop(h).(*routeNode)
		curKey := stateKey{cur.stationID, cur.lineID}

		// skip stale entries
		if d, ok := dist[curKey]; ok && cur.cost > d {
			continue
		}

		// goal: a different station of destKind
		if cur.stationID != fromID && state.Stations[cur.stationID].Kind == destKind {
			goalKey = curKey
			found = true
			break
		}

		// expand neighbours
		for _, nb := range g.Neighbours(cur.stationID) {
			if nb < 0 || nb >= len(state.Stations) || !state.Stations[nb].Alive {
				continue
			}
			for _, lineID := range g.LinesFor(cur.stationID, nb) {
				extra := 0.0
				if cur.lineID != -1 && cur.lineID != lineID {
					extra = transferPenalty
				}
				newCost := cur.cost + 1.0 + extra
				nbKey := stateKey{nb, lineID}
				if d, ok := dist[nbKey]; !ok || newCost < d {
					dist[nbKey] = newCost
					prev[nbKey] = curKey
					heap.Push(h, &routeNode{stationID: nb, lineID: lineID, cost: newCost})
				}
			}
		}
	}

	if !found {
		return RouteInfo{NextLineID: -1}
	}

	// reconstruct path (reversed: goal → start)
	path := make([]stateKey, 0, 8)
	cur := goalKey
	for {
		path = append(path, cur)
		p, ok := prev[cur]
		if !ok {
			break
		}
		cur = p
	}

	// path[len-2] is the first hop from fromID; its lineID is the NextLineID
	nextLineID := -1
	if len(path) >= 2 {
		nextLineID = path[len(path)-2].lineID
	}

	// count transfers by walking the path forward (reversed slice)
	transfers := 0
	prevLine := -1
	for i := len(path) - 1; i >= 0; i-- {
		l := path[i].lineID
		if l == -1 {
			continue
		}
		if prevLine != -1 && l != prevLine {
			transfers++
		}
		prevLine = l
	}

	return RouteInfo{
		Reachable:  true,
		IsDirect:   transfers == 0,
		Transfers:  transfers,
		NextLineID: nextLineID,
		TotalCost:  dist[goalKey],
	}
}

// FindRoute delegates to FindOptimalRoute for backward compatibility.
func FindRoute(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) RouteInfo {
	return FindOptimalRoute(g, state, fromID, destKind)
}

// CanReach returns true if fromID can reach any alive destKind station.
func CanReach(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) bool {
	return FindOptimalRoute(g, state, fromID, destKind).Reachable
}
