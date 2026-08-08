package engine

type Line struct {
	ID       int
	Stations []int
	Corners  []Pos
	Removed  bool
}
