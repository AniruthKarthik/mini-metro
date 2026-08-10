package engine

type Line struct {
	ID        int
	Stations  []int
	Corners   []Pos
	TunnelAt  []bool // TunnelAt[i] = true means segment stations[i]→stations[i+1] used a tunnel token
	Removed   bool
}
