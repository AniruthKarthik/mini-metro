package engine

type ResourcePool struct {
	Lines        int `json:"lines"`
	Trains       int `json:"trains"`
	Tunnels      int `json:"tunnels"`
	Carriages    int `json:"carriages"`
	Interchanges int `json:"interchanges"`
}

func NewResourcePool() ResourcePool {
	return ResourcePool{
		Lines:     1,
		Trains:    1,
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
	case RewardInterchange:
		p.Interchanges++
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
	case RewardInterchange:
		return p.Interchanges > 0
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
	case RewardInterchange:
		p.Interchanges--
	}
	return true
}
