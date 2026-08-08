package engine

type Passenger struct {
	ID          int
	Origin      int
	Destination StationKind
	SpawnTick   uint64
}
