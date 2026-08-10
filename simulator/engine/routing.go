package engine

import (
	"container/heap"
	"math"
)

// RouteInfo is the result of an A* route query from a station to a destination kind.
type RouteInfo struct {
	Reachable     bool    // true if a path exists to any alive destKind station
	IsDirect      bool    // true if reachable with zero line changes
	Transfers     int     // number of line changes in the optimal path
	NextLineID    int     // first line to board; -1 if already at dest or unreachable
	NextDirection int     // +1 or -1 direction to board on NextLineID (0 if unreachable/at dest)
	TotalCost     float64 // Total travel time in seconds of the optimal path
}

// routeNode is an A* priority-queue element indexed on (stationID, lineID, direction).
type routeNode struct {
	stationID int
	lineID    int // -1 = not yet on any line (origin state)
	direction int // 0 = origin state, +1 = forward, -1 = backward
	g         float64
	h         float64
	f         float64
	index     int
}

type routeHeap []*routeNode

func (h routeHeap) Len() int           { return len(h) }
func (h routeHeap) Less(i, j int) bool { return h[i].f < h[j].f }
func (h routeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i]; h[i].index = i; h[j].index = j }
func (h *routeHeap) Push(x interface{}) {
	n := x.(*routeNode)
	n.index = len(*h)
	*h = append(*h, n)
}
func (h *routeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

type stateKey struct {
	stationID int
	lineID    int
	direction int
}

// calcHeuristic computes Euclidean distance to nearest alive destKind station / trainSpeed.
func calcHeuristic(state *GameState, stationID int, destKind StationKind) float64 {
	if stationID < 0 || stationID >= len(state.Stations) {
		return math.MaxFloat64
	}
	stPos := state.Stations[stationID].Pos
	minDist := math.MaxFloat64
	found := false
	for i := range state.Stations {
		st := &state.Stations[i]
		if st.Alive && st.Kind == destKind {
			d := distance(stPos, st.Pos)
			if d < minDist {
				minDist = d
				found = true
			}
		}
	}
	if !found {
		return math.MaxFloat64
	}
	return minDist / trainSpeed
}

// getSegmentDirection determines the direction (+1 or -1) along lineID from u to v.
func getSegmentDirection(line *Line, u, v int) int {
	if line.IsLoop {
		return 1
	}
	uIdx := -1
	vIdx := -1
	for idx, stID := range line.Stations {
		if uIdx == -1 && stID == u {
			uIdx = idx
		}
		if vIdx == -1 && stID == v {
			vIdx = idx
		}
	}
	if uIdx != -1 && vIdx != -1 && vIdx < uIdx {
		return -1
	}
	return 1
}

// expectedTrainWaitTime calculates expected seconds until a train on lineID reaches stationID heading in direction.
func expectedTrainWaitTime(state *GameState, stationID, lineID, direction int) float64 {
	if lineID < 0 || lineID >= len(state.Lines) {
		return 1e6
	}
	line := &state.Lines[lineID]
	if line.Removed || len(line.Stations) < 2 {
		return 1e6
	}

	var active []*Train
	for i := range state.Trains {
		tr := &state.Trains[i]
		if tr.Active && tr.LineID == lineID {
			active = append(active, tr)
		}
	}
	if len(active) == 0 {
		return 1e6 // no active train on this line
	}

	N := len(line.Stations)
	targetIdx := -1
	for idx, stID := range line.Stations {
		if stID == stationID {
			targetIdx = idx
			break
		}
	}
	if targetIdx == -1 {
		return 1e6
	}

	minWait := math.MaxFloat64

	for _, tr := range active {
		// If train is currently at stationID and moving in direction (or direction is 0)
		if tr.Segment >= 0 && tr.Segment < N && line.Stations[tr.Segment] == stationID &&
			(tr.Direction == direction || direction == 0) {
			w := tr.DwellRemaining
			if w < minWait {
				minWait = w
			}
			continue
		}

		// Simulate train movement to targetIdx
		seg := tr.Segment
		prog := tr.Progress
		dir := tr.Direction
		dwell := tr.DwellRemaining
		totalTime := 0.0

		maxSteps := 50
		found := false
		for step := 0; step < maxSteps; step++ {
			// determine next station
			var nextSeg int
			if line.IsLoop {
				nextSeg = (seg + dir + N) % N
			} else {
				nextSeg = seg + dir
				if nextSeg < 0 || nextSeg >= N {
					dir = -dir
					nextSeg = seg + dir
				}
			}

			// travel time for segment seg -> nextSeg
			st1 := line.Stations[seg]
			st2 := line.Stations[nextSeg]
			d := distance(state.Stations[st1].Pos, state.Stations[st2].Pos)
			segTime := d / trainSpeed

			timeOnSeg := (1.0 - prog) * segTime + dwell
			totalTime += timeOnSeg
			prog = 0
			dwell = 0

			seg = nextSeg

			// check if we reached target station with target direction
			if seg == targetIdx && (dir == direction || direction == 0) {
				found = true
				if totalTime < minWait {
					minWait = totalTime
				}
				break
			}

			// add station dwell time
			if state.Stations[line.Stations[seg]].IsInterchange {
				dwell = dwellTime / 2.0
			} else {
				dwell = dwellTime
			}
		}

		if !found && minWait == math.MaxFloat64 {
			// Fallback wait time estimate
			totalTime := float64(N) * 10.0
			if totalTime < minWait {
				minWait = totalTime
			}
		}
	}

	// Add queue capacity delay factor
	queueWait := 0.0
	if stationID >= 0 && stationID < len(state.Stations) {
		qLen := len(state.Stations[stationID].Queue)
		if qLen > 0 {
			queueWait = math.Floor(float64(qLen)/6.0) * 10.0
		}
	}

	return minWait + queueWait
}

// FindOptimalRoute runs A* on the (stationID, lineID, direction) state space.
func FindOptimalRoute(g *NetworkGraph, state *GameState, fromID int, destKind StationKind) RouteInfo {
	if fromID < 0 || fromID >= len(state.Stations) || !state.Stations[fromID].Alive {
		return RouteInfo{NextLineID: -1}
	}
	if state.Stations[fromID].Kind == destKind {
		return RouteInfo{Reachable: true, IsDirect: true, NextLineID: -1}
	}
	if len(g.Adj) == 0 {
		return RouteInfo{NextLineID: -1}
	}

	initH := calcHeuristic(state, fromID, destKind)
	if initH == math.MaxFloat64 {
		return RouteInfo{NextLineID: -1}
	}

	dist := make(map[stateKey]float64)
	prev := make(map[stateKey]stateKey)

	h := &routeHeap{}
	heap.Init(h)

	startKey := stateKey{fromID, -1, 0}
	dist[startKey] = 0
	heap.Push(h, &routeNode{stationID: fromID, lineID: -1, direction: 0, g: 0, h: initH, f: initH})

	var goalKey stateKey
	found := false

	for h.Len() > 0 {
		cur := heap.Pop(h).(*routeNode)
		curKey := stateKey{cur.stationID, cur.lineID, cur.direction}

		// skip stale entries
		if d, ok := dist[curKey]; ok && cur.g > d {
			continue
		}

		// goal check: reached an active station of destKind
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
				if lineID < 0 || lineID >= len(state.Lines) || state.Lines[lineID].Removed {
					continue
				}
				line := &state.Lines[lineID]
				dir := getSegmentDirection(line, cur.stationID, nb)

				// Calculate edge cost: segment ride time + dwell at nb
				segDist := distance(state.Stations[cur.stationID].Pos, state.Stations[nb].Pos)
				rideTime := segDist / trainSpeed
				var dwell float64
				if state.Stations[nb].IsInterchange {
					dwell = dwellTime / 2.0
				} else {
					dwell = dwellTime
				}

				// Wait penalty for embarking or transferring lines / directions
				waitPenalty := 0.0
				if cur.lineID == -1 || cur.lineID != lineID || cur.direction != dir {
					waitPenalty = expectedTrainWaitTime(state, cur.stationID, lineID, dir)
				}

				stepCost := rideTime + dwell + waitPenalty
				newG := cur.g + stepCost
				nbKey := stateKey{nb, lineID, dir}

				if d, ok := dist[nbKey]; !ok || newG < d {
					dist[nbKey] = newG
					prev[nbKey] = curKey
					nbH := calcHeuristic(state, nb, destKind)
					heap.Push(h, &routeNode{
						stationID: nb,
						lineID:    lineID,
						direction: dir,
						g:         newG,
						h:         nbH,
						f:         newG + nbH,
					})
				}
			}
		}
	}

	if !found {
		return RouteInfo{NextLineID: -1}
	}

	// reconstruct path (reversed: goal -> start)
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

	// path[len(path)-2] is the first hop from fromID
	nextLineID := -1
	nextDirection := 0
	if len(path) >= 2 {
		firstHop := path[len(path)-2]
		nextLineID = firstHop.lineID
		nextDirection = firstHop.direction
	}

	// count transfers by walking path forward
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
		Reachable:     true,
		IsDirect:      transfers == 0,
		Transfers:     transfers,
		NextLineID:    nextLineID,
		NextDirection: nextDirection,
		TotalCost:     dist[goalKey],
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

