package main

import (
	"fmt"
	"strings"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func kindName(k engine.StationKind) string {
	switch k {
	case engine.Circle:
		return "Circle"
	case engine.Triangle:
		return "Triangle"
	case engine.Square:
		return "Square"
	case engine.Star:
		return "Star"
	case engine.Pentagon:
		return "Pentagon"
	default:
		return fmt.Sprintf("Unknown(%d)", k)
	}
}

func sep() { fmt.Println(strings.Repeat("-", 60)) }

func header(title string) {
	sep()
	fmt.Printf("  %s\n", title)
	sep()
}

func pass(msg string) { fmt.Printf("  [PASS] %s\n", msg) }
func fail(msg string) { fmt.Printf("  [FAIL] %s\n", msg) }

func check(cond bool, msg string) {
	if cond {
		pass(msg)
	} else {
		fail(msg)
	}
}

func main() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("  MINI METRO SIMULATOR - FULL INTEGRATION TEST")
	fmt.Println(strings.Repeat("=", 60))

	// --- 1. Resource pool ---
	header("1. Resource Pool")
	stations := []engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Star, Pos: engine.Pos{X: 30, Y: 0}},
	}
	sim := engine.NewSimulator(stations)
	pool := sim.State.Resources
	check(pool.Lines == 3, fmt.Sprintf("initial lines = %d (want 3)", pool.Lines))
	check(pool.Trains == 3, fmt.Sprintf("initial trains = %d (want 3)", pool.Trains))
	check(pool.Tunnels == 0, fmt.Sprintf("initial tunnels = %d (want 0)", pool.Tunnels))
	check(pool.Carriages == 0, fmt.Sprintf("initial carriages = %d (want 0)", pool.Carriages))

	// --- 2. Gate actions on resource limits ---
	header("2. Action Gating on Resources")
	// Spend all 3 lines
	sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	sim.ApplyAction(engine.AddLine{Stations: []int{1, 2}})
	sim.ApplyAction(engine.AddLine{Stations: []int{2, 3}})
	check(sim.State.Resources.Lines == 0, "all 3 lines spent")
	err := sim.ApplyAction(engine.AddLine{Stations: []int{0, 2}})
	check(err != nil, "4th line rejected when pool empty")

	// Spend all 3 trains
	sim.ApplyAction(engine.AddTrain{LineID: 0})
	sim.ApplyAction(engine.AddTrain{LineID: 1})
	sim.ApplyAction(engine.AddTrain{LineID: 2})
	check(sim.State.Resources.Trains == 0, "all 3 trains spent")
	err = sim.ApplyAction(engine.AddTrain{LineID: 0})
	check(err != nil, "4th train rejected when pool empty")

	// No carriages yet
	err = sim.ApplyAction(engine.AddCarriage{TrainID: 0})
	check(err != nil, "carriage rejected when pool empty")

	// --- 3. Resource return on line removal ---
	header("3. Resource Return on Removal")
	prevLines := sim.State.Resources.Lines
	prevTrains := sim.State.Resources.Trains
	sim.ApplyAction(engine.RemoveLine{LineID: 0})
	check(sim.State.Resources.Lines == prevLines+1, "line refunded on removal")
	check(sim.State.Resources.Trains == prevTrains+1, "train refunded on line removal")
	check(!sim.State.Trains[0].Active, "train deactivated on line removal")

	// --- 4. Adjacency graph ---
	header("4. Adjacency Graph (network topology)")
	// Rebuild graph by stepping once
	sim.Step(0.01)
	adj := sim.State.Graph.Adj
	// Line 1: 1-2, Line 2: 2-3 (Line 0 was removed)
	check(len(adj) > 0, "adjacency graph populated after step")
	check(contains(adj[1], 2), "station 1 neighbours station 2")
	check(contains(adj[2], 3), "station 2 neighbours station 3")
	check(!contains(adj[0], 1), "station 0 not connected (line 0 removed)")

	// TopologyVersion bumps
	vBefore := sim.State.TopologyVersion
	sim.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	check(sim.State.TopologyVersion == vBefore+1, "TopologyVersion incremented on AddLine")

	// --- 5. Passenger route checking ---
	header("5. Passenger Route Checking")
	// Fresh simulator: line 0→1→2, route from 0 to Triangle (1) should be direct
	sim2 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	sim2.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})
	sim2.Step(0.01) // trigger graph rebuild
	r := engine.FindRoute(&sim2.State.Graph, &sim2.State, 0, engine.Triangle)
	check(r.Reachable, "station 0 can reach Triangle")
	check(r.IsDirect, "route is direct (same line)")

	// Route to a kind not on the network
	r2 := engine.FindRoute(&sim2.State.Graph, &sim2.State, 0, engine.Pentagon)
	check(!r2.Reachable, "station 0 cannot reach Pentagon (no such station)")

	// Transfer route: line A: 0-1, line B: 1-2. From 0 to Square(2) is a transfer.
	sim3 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	sim3.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	sim3.ApplyAction(engine.AddLine{Stations: []int{1, 2}})
	sim3.Step(0.01)
	r3 := engine.FindRoute(&sim3.State.Graph, &sim3.State, 0, engine.Square)
	check(r3.Reachable, "transfer route: 0→1→2 reachable")
	check(!r3.IsDirect, "transfer route: not direct (two lines)")

	// --- 6. Boarding uses network pathing ---
	header("6. Boarding Filtered by Route Reachability")
	// Station 0 (Circle) and station 1 (Triangle) on line 0. Pentagon has no station.
	// Passenger wanting Pentagon should stay in queue.
	sim4 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})
	sim4.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	sim4.ApplyAction(engine.AddTrain{LineID: 0})
	// Manually inject a passenger wanting Pentagon (unreachable) into station 0
	sim4.State.Stations[0].Queue = append(sim4.State.Stations[0].Queue, engine.Passenger{
		Origin:      0,
		Destination: engine.Pentagon,
		SpawnTick:   0,
	})
	qBefore := len(sim4.State.Stations[0].Queue)
	sim4.Step(0.5) // enough progress to trigger boardAndAlight
	qAfter := len(sim4.State.Stations[0].Queue)
	check(qBefore == 1, "queue seeded with 1 unreachable passenger")
	// the passenger may or may not have been processed depending on train position,
	// but we verify the train didn't pick up an unreachable passenger
	trainLoad := len(sim4.State.Trains[0].Passengers)
	check(trainLoad == 0 || qAfter > 0, "unreachable passenger not boarded or remains in queue")

	// --- 7. Event scheduler ---
	header("7. Event Scheduler")
	sim5 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
	})
	check(len(sim5.State.Scheduler.Events) == 2, "two events scheduled at start (reward + spawn)")
	spawnEv := sim5.State.Scheduler.Events[0]  // spawn fires at 200, sorts first
	rewardEv := sim5.State.Scheduler.Events[1] // reward fires at 500
	check(spawnEv.FireAt == 200, fmt.Sprintf("spawn fires at tick 200 (got %d)", spawnEv.FireAt))
	check(rewardEv.FireAt == 500, fmt.Sprintf("reward fires at tick 500 (got %d)", rewardEv.FireAt))

	// --- 8. Dynamic station spawning ---
	header("8. Dynamic Station Spawning")
	sim6 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 50, Y: 50}},
	})
	initialCount := len(sim6.State.Stations)
	// Run 201 ticks to cross the first spawn interval
	for i := 0; i < 201; i++ {
		sim6.Step(0.01)
	}
	newCount := len(sim6.State.Stations)
	check(newCount > initialCount, fmt.Sprintf("stations grew from %d to %d after 201 ticks", initialCount, newCount))
	// Check spawned station has a valid kind
	if newCount > initialCount {
		spawned := sim6.State.Stations[initialCount]
		validKind := spawned.Kind >= engine.Circle && spawned.Kind <= engine.Pentagon
		check(validKind, fmt.Sprintf("spawned station kind = %s", kindName(spawned.Kind)))
		check(spawned.Alive, "spawned station is alive")
		check(spawned.ID == initialCount, fmt.Sprintf("spawned station ID = %d", spawned.ID))
	}

	// --- 9. Weekly reward event ---
	header("9. Weekly Reward Event")
	sim7 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
	})
	trainsBefore := sim7.State.Resources.Trains
	// Run 501 ticks to fire the reward
	for i := 0; i < 501; i++ {
		if len(sim7.State.PendingRewardChoices) > 0 {
			// auto-accept first choice
			sim7.ApplyAction(engine.ChooseReward{Choice: sim7.State.PendingRewardChoices[0]})
		}
		sim7.Step(0.01)
	}
	check(sim7.State.Resources.Trains > trainsBefore, fmt.Sprintf("trains increased from %d (reward granted)", trainsBefore))
	// Next reward event should be rescheduled
	hasReward := false
	for _, ev := range sim7.State.Scheduler.Events {
		if ev.Kind == 0 { // EventReward = 0
			hasReward = true
		}
	}
	check(hasReward, "reward event rescheduled after firing")

	// --- Summary ---
	fmt.Println()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("  ALL CHECKS COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
}

// contains is a local helper for the test runner.
func contains(slice []int, val int) bool {
	for _, x := range slice {
		if x == val {
			return true
		}
	}
	return false
}
