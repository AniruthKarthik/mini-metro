package engine

type Passenger struct {
	ID          int
	Destination StationKind
	SpawnTick   uint64
}
