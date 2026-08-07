package engine

type pos struct {
	x float64
	y float64
}

type Station struct {
	ID    int
	Kind  int
	Pos   pos
	Queue []int
}

type Line struct {
	ID       int
	Stations []int
	Corners  []pos
}

type Train struct {
	ID         int
	Line       Line
	Segment    int // line between stations, explains position of the train in the series of lines
	Passengers []float64
}

type GameState struct {
	Stations []Station
	Lines    []Line
	Trains   []Train

	Score int
	Tick  uint64
	Alive bool
}
