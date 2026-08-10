package engine

type StationKind int

const (
	Circle StationKind = iota
	Triangle
	Square
	Star
	Pentagon
)

type Station struct {
	ID                int
	Kind              StationKind
	Pos               Pos
	Queue             []Passenger
	Capacity          int     // max queue size before overcrowding timer starts
	Alive             bool
	IsInterchange     bool
	OvercrowdingTimer float64 // ticks remaining before game over; -1 = no active timer
}

const defaultStationCapacity = 6
