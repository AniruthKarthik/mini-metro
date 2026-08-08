package engine

type ResourcePool struct {
	Lines     int
	Trains    int
	Tunnels   int
	Carriages int
}

func NewResourcePool() ResourcePool {
	return ResourcePool{
		Lines:     3,
		Trains:    3,
		Tunnels:   0,
		Carriages: 0,
	}
}

type RewardType int

const (
	RewardLine RewardType = iota
	RewardTrain
	RewardTunnel
	RewardCarriage
	RewardInterchange // upgrades a station to a hub, no pool count needed
)

func (p *ResourcePool) Grant(r RewardType) {
	switch r {
	case RewardLine:
		p.Lines++
	case RewardTrain:
		p.Trains++
	case RewardTunnel:
		p.Tunnels++
	case RewardCarriage:
		p.Carriages++
	}
}

func (p *ResourcePool) CanSpend(r RewardType) bool {
	switch r {
	case RewardLine:
		return p.Lines > 0
	case RewardTrain:
		return p.Trains > 0
	case RewardTunnel:
		return p.Tunnels > 0
	case RewardCarriage:
		return p.Carriages > 0
	}
	return false
}

func (p *ResourcePool) Spend(r RewardType) bool {
	if !p.CanSpend(r) {
		return false
	}
	switch r {
	case RewardLine:
		p.Lines--
	case RewardTrain:
		p.Trains--
	case RewardTunnel:
		p.Tunnels--
	case RewardCarriage:
		p.Carriages--
	}
	return true
}
