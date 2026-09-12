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

// progressiveStationWeights computes station spawn weights based on game progression.
// Early game (< 7 stations): strictly base shapes (Circle, Triangle, Square).
// Mid game (7-9 stations): uncommon shapes (Star, Pentagon) unlock as unique landmarks.
// Late game (>= 10 stations): rare unique shapes (Cross/plus, Gem, Sector, Drop, Oval) unlock.
// Uncommon and rare shapes only spawn if not already present on the active map.
func (s *Simulator) progressiveStationWeights() map[StationKind]int {
	numStations := len(s.State.Stations)
	existingKinds := make(map[StationKind]int)
	for i := range s.State.Stations {
		if s.State.Stations[i].Alive {
			existingKinds[s.State.Stations[i].Kind]++
		}
	}

	w := map[StationKind]int{
		Circle:   10,
		Triangle: 8,
		Square:   6,
	}

	// Early game (< 7 stations): strictly base shapes (Circle, Triangle, Square)
	if numStations < 7 {
		return w
	}

	// Mid game (7-9 stations): Star and Pentagon unlock as unique landmarks
	if existingKinds[Star] == 0 {
		w[Star] = 2
	}
	if existingKinds[Pentagon] == 0 {
		w[Pentagon] = 2
	}

	// Late game (>= 10 stations): Cross (plus), Gem, Sector, Drop, Oval unlock as unique landmarks
	if numStations >= 10 {
		rareKinds := []StationKind{Cross, Gem, Sector, Drop, Oval}
		for _, k := range rareKinds {
			if existingKinds[k] == 0 {
				w[k] = 1
			}
		}
	}

	return w
}

// weightedRandomKind returns a StationKind sampled proportionally to progressive weights or custom weights.
func (s *Simulator) weightedRandomKind(rng *rand.Rand) StationKind {
	weights := s.State.StationWeights
	if weights == nil {
		weights = s.progressiveStationWeights()
	}
	total := 0
	for _, w := range weights {
		if w > 0 {
			total += w
		}
	}
	if total <= 0 {
		return Circle
	}
	r := rng.Intn(total)
	allKinds := []StationKind{Circle, Triangle, Square, Star, Pentagon, Gem, Sector, Cross, Drop, Oval}
	for _, kind := range allKinds {
		w, ok := weights[kind]
		if !ok || w <= 0 {
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
