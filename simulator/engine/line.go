package engine

type Line struct {
	ID         int
	Stations   []int
	Corners    []Pos
	TunnelAt   []bool // TunnelAt[i] = true means segment stations[i]→stations[i+1] used a tunnel token
	IsLoop     bool   // IsLoop = true means last station wraps back to first, trains travel one-way
	LoopTunnel bool   // LoopTunnel = true means the wrap-around segment used a tunnel token
	Removed    bool
}
