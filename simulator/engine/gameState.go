package engine

type GameState struct {
	Stations []Station
	Lines    []Line
	Trains   []Train

	Score int
	Tick  uint64
	Alive bool
}
