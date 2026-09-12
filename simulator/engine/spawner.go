package engine

import (
	"math/rand"
)

// spawnInterval returns how many fixed 30 Hz ticks between automatic station spawns.
func spawnInterval() uint64 { return 900 }

func initialSpawnInterval() uint64 { return 900 }

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

// SetStationSpawnWeights configures custom station spawn weights on the simulator instance.
func (s *Simulator) SetStationSpawnWeights(weights map[StationKind]int) {
	s.State.StationWeights = weights
}

// ResetStationSpawnWeights clears custom weights, reverting to default stationWeights.
func (s *Simulator) ResetStationSpawnWeights() {
	s.State.StationWeights = nil
}

// weightedRandomKind returns a StationKind sampled proportionally to stationWeights or custom weights.
func (s *Simulator) weightedRandomKind(rng *rand.Rand) StationKind {
	weights := s.State.StationWeights
	if weights == nil {
		weights = stationWeights
	}
	total := 0
	for _, w := range weights {
		total += w
	}
	if total <= 0 {
		return Circle
	}
	r := rng.Intn(total)
	allKinds := []StationKind{Circle, Triangle, Square, Star, Pentagon, Gem, Sector, Cross, Drop, Oval}
	for _, kind := range allKinds {
		w, ok := weights[kind]
		if !ok {
			continue
		}
		r -= w
		if r < 0 {
			return kind
		}
	}
	return Circle
}

// spawnStation appends a new alive station with a weighted random kind and schedules the next spawn.
func (s *Simulator) spawnStation() {
	id := len(s.State.Stations)

	var spawnPos Pos
	found := false
	const minDist = 12.0

	for attempt := 0; attempt < 100; attempt++ {
		cand := Pos{
			X: 12.0 + s.RNG().Float64()*76.0,
			Y: 12.0 + s.RNG().Float64()*76.0,
		}

		if PosInWater(cand, s.State.Rivers, s.State.WaterPolygons, 4.0) {
			continue
		}

		tooClose := false
		for i := range s.State.Stations {
			if s.State.Stations[i].Alive && distance(cand, s.State.Stations[i].Pos) < minDist {
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
		spawnPos = Pos{X: 15.0 + s.RNG().Float64()*70.0, Y: 15.0 + s.RNG().Float64()*70.0}
	}

	s.State.Stations = append(s.State.Stations, Station{
		ID:                id,
		Kind:              s.weightedRandomKind(s.RNG()),
		Pos:               spawnPos,
		Capacity:          defaultStationCapacity,
		Alive:             true,
		OvercrowdingTimer: -1,
	})
	s.State.Scheduler.Schedule(s.State.Tick+spawnInterval(), EventSpawnStation)
}
