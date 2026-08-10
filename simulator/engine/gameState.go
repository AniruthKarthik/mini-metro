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
	Score                int
	Tick                 uint64
	Alive                bool
	MaxTrainsPerLine     int // max trains allowed per line (default 4)
}

