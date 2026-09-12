package engine

type GameState struct {
	MapName              string
	Stations             []Station
	Lines                []Line
	Trains               []Train
	Rivers               []RiverSegment
	WaterPolygons        []WaterPolygon
	Resources            ResourcePool
	Graph                NetworkGraph // cached adjacency graph; rebuilt when TopologyVersion changes
	Scheduler            EventScheduler
	PendingRewardChoices []RewardType
	TopologyVersion      uint64
	NextPassengerID      int
	Score                int
	Tick                 uint64
	GameTimeSeconds      float64
	Alive                bool
	MaxTrainsPerLine     int                 // max trains allowed per line (default 4)
	StationWeights       map[StationKind]int // custom station spawn weights; if nil, uses default stationWeights
}
