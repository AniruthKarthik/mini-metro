package engine

import "math/rand"

// spawnInterval returns how many ticks between automatic station spawns.
func spawnInterval() uint64 { return 200 }

// stationWeights is the relative spawn probability for each StationKind.
var stationWeights = map[StationKind]int{
	Circle:   5,
	Triangle: 4,
	Square:   3,
	Star:     2,
	Pentagon: 1,
}

// weightedRandomKind returns a StationKind sampled proportionally to stationWeights.
func weightedRandomKind() StationKind {
	total := 0
	for _, w := range stationWeights {
		total += w
	}
	r := rand.Intn(total)
	for _, kind := range []StationKind{Circle, Triangle, Square, Star, Pentagon} {
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
		ID:    id,
		Kind:  weightedRandomKind(),
		Pos:   Pos{X: rand.Float64() * 100.0, Y: rand.Float64() * 100.0},
		Alive: true,
	})
	s.State.Scheduler.Schedule(s.State.Tick+spawnInterval(), EventSpawnStation)
}
