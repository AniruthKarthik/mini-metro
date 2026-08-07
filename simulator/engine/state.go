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
	Alive bool
}

type Line struct {
	ID       int
	Stations []int
	Corners  []pos
	Removed  bool
}

type Train struct {
	ID         int
	LineID     int
	Segment    int // line between stations, explains position of the train in the series of lines
	Progress   float64
	Direction  int
	Capacity   int
	Passengers []int
	Active     bool
}

type GameState struct {
	Stations []Station
	Lines    []Line
	Trains   []Train

	Score int
	Tick  uint64
	Alive bool
}
