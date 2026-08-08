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
	ID    int
	Kind  StationKind
	Pos   Pos
	Queue []Passenger
	Alive bool
}
