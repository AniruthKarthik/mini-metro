package engine

import (
	"testing"
)

func TestProgressiveStationWeights(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	// 1. Early Game (< 7 stations)
	// London starts with 3 stations (Circle, Triangle, Square)
	if len(sim.State.Stations) != 3 {
		t.Fatalf("expected 3 starter stations, got %d", len(sim.State.Stations))
	}

	wEarly := sim.progressiveStationWeights()
	if wEarly[Circle] <= 0 || wEarly[Triangle] <= 0 || wEarly[Square] <= 0 {
		t.Errorf("expected positive weights for Circle, Triangle, Square in early game, got: %+v", wEarly)
	}
	complexKinds := []StationKind{Star, Pentagon, Gem, Sector, Cross, Drop, Oval}
	for _, k := range complexKinds {
		if wEarly[k] > 0 {
			t.Errorf("expected 0 weight for complex kind %s in early game (< 7 stations), got %d", k, wEarly[k])
		}
	}

	// 2. Early Game Dynamic Spawns (Stations 3 through 6)
	for i := len(sim.State.Stations); i < 6; i++ {
		sim.spawnStation()
		newSt := sim.State.Stations[len(sim.State.Stations)-1]
		if newSt.Kind != Circle && newSt.Kind != Triangle && newSt.Kind != Square {
			t.Fatalf("station %d spawned complex kind %s in early game; expected base shape", i, newSt.Kind)
		}
	}

	// 3. Mid Game (7 - 9 stations): Star and Pentagon unlock
	for len(sim.State.Stations) < 7 {
		sim.spawnStation()
	}
	wMid := sim.progressiveStationWeights()
	if wMid[Star] <= 0 || wMid[Pentagon] <= 0 {
		t.Errorf("expected Star and Pentagon to unlock in mid game (>= 7 stations), got: %+v", wMid)
	}
	rareKinds := []StationKind{Gem, Sector, Cross, Drop, Oval}
	for _, k := range rareKinds {
		if wMid[k] > 0 {
			t.Errorf("expected 0 weight for rare kind %s in mid game (< 10 stations), got %d", k, wMid[k])
		}
	}

	// 4. Late Game (>= 10 stations): Cross (plus), Gem, etc. unlock
	for len(sim.State.Stations) < 10 {
		sim.spawnStation()
	}
	wLate := sim.progressiveStationWeights()
	if wLate[Cross] <= 0 || wLate[Gem] <= 0 {
		t.Errorf("expected Cross (plus) and Gem to unlock in late game (>= 10 stations), got: %+v", wLate)
	}

	// 5. Unique Landmark Invariant
	// If a Star is already active, its weight drops to 0
	sim.State.Stations = append(sim.State.Stations, Station{
		ID:    len(sim.State.Stations),
		Kind:  Star,
		Alive: true,
	})
	wWithStar := sim.progressiveStationWeights()
	if wWithStar[Star] != 0 {
		t.Errorf("expected 0 weight for Star once already present on map, got %d", wWithStar[Star])
	}

	// If a Cross is already active, its weight drops to 0
	sim.State.Stations = append(sim.State.Stations, Station{
		ID:    len(sim.State.Stations),
		Kind:  Cross,
		Alive: true,
	})
	wWithCross := sim.progressiveStationWeights()
	if wWithCross[Cross] != 0 {
		t.Errorf("expected 0 weight for Cross once already present on map, got %d", wWithCross[Cross])
	}

	// 6. Custom Weights Override
	custom := map[StationKind]int{Circle: 5, Star: 5}
	sim.SetStationSpawnWeights(custom)
	kind := sim.weightedRandomKind(sim.RNG())
	if kind != Circle && kind != Star {
		t.Errorf("expected custom weights to restrict kind to Circle or Star, got %s", kind)
	}
	sim.ResetStationSpawnWeights()
}
