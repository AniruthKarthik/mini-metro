package engine

type StationKind int

const (
	Circle StationKind = iota
	Triangle
	Square
	Star
	Pentagon
	Gem
	Sector
	Cross
	Drop
	Oval
)

func (k StationKind) String() string {
	switch k {
	case Circle:
		return "Circle"
	case Triangle:
		return "Triangle"
	case Square:
		return "Square"
	case Star:
		return "Star"
	case Pentagon:
		return "Pentagon"
	case Gem:
		return "Gem"
	case Sector:
		return "Sector"
	case Cross:
		return "Cross"
	case Drop:
		return "Drop"
	case Oval:
		return "Oval"
	default:
		return "Unknown"
	}
}

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
