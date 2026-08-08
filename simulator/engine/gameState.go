package engine

type GameState struct {
	Stations []Station
	Lines    []Line
	Trains   []Train
	TopologyVersion uint64
	Score int
	Tick  uint64
	Alive bool
}
