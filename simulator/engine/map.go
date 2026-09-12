package engine

import (
	"math/rand"
)

type MapConfig struct {
	Name                     string
	MaxLines                 int
	MaxTrainsPerLine         int
	InitialResources         ResourcePool
	Rivers                   []RiverSegment
	WaterPolygons            []WaterPolygon
	InitialStations          []Station
	RandomizeInitialStations bool
}

// LondonMap returns a MapConfig for London with the River Thames.
func LondonMap() MapConfig {
	return MapConfig{
		Name:             "London",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     3,
			Trains:    3,
			Tunnels:   3,
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
		RandomizeInitialStations: true,
	}
}

// NYCMap returns a MapConfig for New York City with Hudson and East Rivers.
func NYCMap() MapConfig {
	return MapConfig{
		Name:             "New York City",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     3,
			Trains:    3,
			Tunnels:   3,
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
		RandomizeInitialStations: true,
	}
}

// TokyoMap returns a MapConfig for Tokyo with Sumida River and Tokyo Bay.
func TokyoMap() MapConfig {
	return MapConfig{
		Name:             "Tokyo",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     3,
			Trains:    3,
			Tunnels:   3,
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
		RandomizeInitialStations: true,
	}
}

// BerlinMap returns a MapConfig for Berlin with no water bodies and no tunnels.
// Station positions are randomized across the open plain layout.
func BerlinMap() MapConfig {
	return MapConfig{
		Name:             "Berlin",
		MaxLines:         7,
		MaxTrainsPerLine: 4,
		InitialResources: ResourcePool{
			Lines:     3,
			Trains:    3,
			Tunnels:   0,
			Carriages: 0,
		},
		InitialStations: []Station{
			{ID: 0, Kind: Circle, Pos: Pos{X: 20, Y: 25}},
			{ID: 1, Kind: Triangle, Pos: Pos{X: 50, Y: 60}},
			{ID: 2, Kind: Square, Pos: Pos{X: 80, Y: 25}},
		},
		RandomizeInitialStations: true,
	}
}

// RandomizeInitialStations randomizes the positions of the initial starter stations
// while strictly preserving each station's original Kind and ID (Station 0: Circle,
// Station 1: Triangle, Station 2: Square). Placed positions avoid water bodies and
// maintain minimum spacing between each other.
func (s *Simulator) RandomizeInitialStations() {
	if len(s.State.Stations) == 0 {
		// Fallback: create the canonical 3 starter stations
		s.State.Stations = []Station{
			{ID: 0, Kind: Circle, Capacity: defaultStationCapacity, Alive: true, OvercrowdingTimer: -1},
			{ID: 1, Kind: Triangle, Capacity: defaultStationCapacity, Alive: true, OvercrowdingTimer: -1},
			{ID: 2, Kind: Square, Capacity: defaultStationCapacity, Alive: true, OvercrowdingTimer: -1},
		}
	}

	placed := make([]Pos, 0, len(s.State.Stations))
	const minDist = 18.0

	for i := range s.State.Stations {
		var spawnPos Pos
		found := false

		for attempt := 0; attempt < 200; attempt++ {
			cand := Pos{
				X: 18.0 + s.RNG().Float64()*64.0,
				Y: 18.0 + s.RNG().Float64()*64.0,
			}

			if PosInWater(cand, s.State.Rivers, s.State.WaterPolygons, 4.0) {
				continue
			}

			tooClose := false
			for _, pos := range placed {
				if distance(cand, pos) < minDist {
					tooClose = true
					break
				}
			}

			if !tooClose {
				spawnPos = cand
				found = true
				break
			}
		}

		if !found {
			// Fallback with fixed spacing if tight space
			spawnPos = Pos{X: 20.0 + float64(i)*25.0, Y: 30.0 + float64(i%2)*30.0}
		}

		s.State.Stations[i].Pos = spawnPos
		placed = append(placed, spawnPos)
	}

	s.rebuildGraphIfNeeded()
}

// NewSimulatorWithMap creates a Simulator configured for a specific MapConfig.
func NewSimulatorWithMap(cfg MapConfig, seed ...uint64) *Simulator {
	stations := make([]Station, len(cfg.InitialStations))
	copy(stations, cfg.InitialStations)
	for i := range stations {
		stations[i].Alive = true
		stations[i].OvercrowdingTimer = -1
		if stations[i].Capacity == 0 {
			stations[i].Capacity = defaultStationCapacity
		}
	}
	var sSeed uint64 = 42
	if len(seed) > 0 {
		sSeed = seed[0]
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
		rng:           rand.New(rand.NewSource(int64(sSeed))),
		ScoringConfig: DefaultScoringConfig(),
	}
	if sim.State.MaxTrainsPerLine <= 0 {
		sim.State.MaxTrainsPerLine = 4
	}

	if cfg.RandomizeInitialStations || len(cfg.InitialStations) == 0 {
		sim.RandomizeInitialStations()
	}

	sim.State.Scheduler.Schedule(rewardInterval(), EventReward)
	sim.State.Scheduler.Schedule(initialSpawnInterval(), EventSpawnStation)
	return sim
}
