package engine

type MapConfig struct {
	Name             string
	MaxLines         int
	MaxTrainsPerLine int
	InitialResources ResourcePool
	Rivers           []RiverSegment
	WaterPolygons    []WaterPolygon
	InitialStations  []Station
}

// LondonMap returns a MapConfig for London with the River Thames.
func LondonMap() MapConfig {
	return MapConfig{
		Name:             "London",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     1,
			Trains:    1,
			Tunnels:   1,
			Carriages: 0,
		},
		Rivers: []RiverSegment{
			// River Thames curving across the map
			{From: Pos{X: 0, Y: 40}, To: Pos{X: 30, Y: 45}, Width: 6.0},
			{From: Pos{X: 30, Y: 45}, To: Pos{X: 60, Y: 35}, Width: 6.0},
			{From: Pos{X: 60, Y: 35}, To: Pos{X: 100, Y: 50}, Width: 6.0},
		},
		InitialStations: []Station{
			{ID: 0, Kind: Circle, Pos: Pos{X: 20, Y: 25}},
			{ID: 1, Kind: Triangle, Pos: Pos{X: 50, Y: 60}},
			{ID: 2, Kind: Square, Pos: Pos{X: 80, Y: 25}},
		},
	}
}

// NYCMap returns a MapConfig for New York City with Hudson and East Rivers.
func NYCMap() MapConfig {
	return MapConfig{
		Name:             "New York City",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     1,
			Trains:    1,
			Tunnels:   2,
			Carriages: 0,
		},
		Rivers: []RiverSegment{
			// Hudson River (left channel)
			{From: Pos{X: 30, Y: 0}, To: Pos{X: 30, Y: 100}, Width: 8.0},
			// East River (right channel)
			{From: Pos{X: 65, Y: 0}, To: Pos{X: 65, Y: 100}, Width: 6.0},
		},
		WaterPolygons: []WaterPolygon{
			// Upper New York Bay at bottom
			{Vertices: []Pos{
				{X: 20, Y: 0}, {X: 80, Y: 0}, {X: 80, Y: 20}, {X: 20, Y: 20},
			}},
		},
		InitialStations: []Station{
			{ID: 0, Kind: Circle, Pos: Pos{X: 15, Y: 50}},   // New Jersey
			{ID: 1, Kind: Triangle, Pos: Pos{X: 48, Y: 50}}, // Manhattan
			{ID: 2, Kind: Square, Pos: Pos{X: 80, Y: 50}},   // Brooklyn/Queens
		},
	}
}

// TokyoMap returns a MapConfig for Tokyo with Sumida River and Tokyo Bay.
func TokyoMap() MapConfig {
	return MapConfig{
		Name:             "Tokyo",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     1,
			Trains:    1,
			Tunnels:   1,
			Carriages: 0,
		},
		Rivers: []RiverSegment{
			// Sumida River
			{From: Pos{X: 50, Y: 100}, To: Pos{X: 60, Y: 40}, Width: 5.0},
		},
		WaterPolygons: []WaterPolygon{
			// Tokyo Bay at bottom right
			{Vertices: []Pos{
				{X: 40, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 40}, {X: 50, Y: 30},
			}},
		},
		InitialStations: []Station{
			{ID: 0, Kind: Circle, Pos: Pos{X: 25, Y: 60}},   // Shinjuku area
			{ID: 1, Kind: Triangle, Pos: Pos{X: 70, Y: 70}}, // Ueno/Asakusa
			{ID: 2, Kind: Square, Pos: Pos{X: 40, Y: 20}},   // Shinagawa/Tokyo
		},
	}
}

// NewSimulatorWithMap creates a Simulator configured for a specific MapConfig.
func NewSimulatorWithMap(cfg MapConfig) *Simulator {
	stations := make([]Station, len(cfg.InitialStations))
	copy(stations, cfg.InitialStations)
	for i := range stations {
		stations[i].Alive = true
		stations[i].OvercrowdingTimer = -1
		if stations[i].Capacity == 0 {
			stations[i].Capacity = defaultStationCapacity
		}
	}
	sim := &Simulator{
		State: GameState{
			MapName:          cfg.Name,
			Stations:         stations,
			Lines:            []Line{},
			Trains:           []Train{},
			Rivers:           append([]RiverSegment(nil), cfg.Rivers...),
			WaterPolygons:    append([]WaterPolygon(nil), cfg.WaterPolygons...),
			Resources:        cfg.InitialResources,
			Score:            0,
			Tick:             0,
			Alive:            true,
			MaxTrainsPerLine: cfg.MaxTrainsPerLine,
		},
	}
	if sim.State.MaxTrainsPerLine <= 0 {
		sim.State.MaxTrainsPerLine = 4
	}
	sim.State.Scheduler.Schedule(rewardInterval(), EventReward)
	sim.State.Scheduler.Schedule(initialSpawnInterval(), EventSpawnStation)
	return sim
}
