package engine

import "math/rand"

// spawnInterval returns how many ticks between automatic station spawns.
func spawnInterval() uint64 { return 200 }

// stationWeights is the relative spawn probability for each StationKind.
var stationWeights = map[StationKind]int{
	Circle:   10,
	Triangle: 8,
	Square:   6,
	Star:     2,
	Pentagon: 2,
	Gem:      1,
	Sector:   1,
	Cross:    1,
	Drop:     1,
	Oval:     1,
}

// weightedRandomKind returns a StationKind sampled proportionally to stationWeights.
func weightedRandomKind() StationKind {
	total := 0
	for _, w := range stationWeights {
		total += w
	}
	r := rand.Intn(total)
	allKinds := []StationKind{Circle, Triangle, Square, Star, Pentagon, Gem, Sector, Cross, Drop, Oval}
	for _, kind := range allKinds {
		r -= stationWeights[kind]
		if r < 0 {
			return kind
		}
	}
	return Circle
}

// spawnStation appends a new alive station with a weighted random kind and schedules the next spawn.
func (s *Simulator) spawnStation() {
	id := len(s.State.Stations)
	s.State.Stations = append(s.State.Stations, Station{
		ID:                id,
		Kind:              weightedRandomKind(),
		Pos:               Pos{X: rand.Float64() * 100.0, Y: rand.Float64() * 100.0},
		Capacity:          defaultStationCapacity,
		Alive:             true,
		OvercrowdingTimer: -1,
	})
	s.State.Scheduler.Schedule(s.State.Tick+spawnInterval(), EventSpawnStation)
}
