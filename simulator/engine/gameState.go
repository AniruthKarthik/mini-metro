package engine

type GameState struct {
	Stations             []Station
	Lines                []Line
	Trains               []Train
	Resources            ResourcePool
	NextRewardAt         float64
	PendingRewardChoices []RewardType
	TopologyVersion      uint64
	Score                int
	Tick                 uint64
	Alive                bool
}
