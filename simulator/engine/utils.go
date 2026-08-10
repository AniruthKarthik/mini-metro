package engine

import "math"

type Pos struct {
	X float64
	Y float64
}

// tunnelDistanceThreshold is the minimum Euclidean distance between stations that requires a tunnel token.
const tunnelDistanceThreshold = 30.0

// distance returns the Euclidean distance between two positions.
func distance(a, b Pos) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func rewardInterval() uint64 {
	return 500
}
