package main

import (
	"fmt"
	"strings"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func kindName(k engine.StationKind) string {
	return k.String()
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

	// --- 10. True A* Mini Metro Routing Verification ---
	header("10. True A* Mini Metro Routing (Direction, Transfers, Travel Time)")
	
	// Test Direction-Aware Boarding
	sim8 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	sim8.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})
	sim8.Step(0.01)

	rDir := engine.FindRoute(&sim8.State.Graph, &sim8.State, 1, engine.Circle)
	check(rDir.Reachable, "station 1 can reach Circle")
	check(rDir.NextLineID == 0, "next line is Line 0")
	check(rDir.NextDirection == -1, "next direction is -1 (towards Station 0)")

	// Seed Station 1 queue with passenger wanting Circle
	sim8.State.Stations[1].Queue = append(sim8.State.Stations[1].Queue, engine.Passenger{
		Origin:      1,
		Destination: engine.Circle,
	})

	// Train on Line 0 at Station 1 heading +1 (away from Circle)
	sim8.State.Trains = append(sim8.State.Trains, engine.Train{
		ID:          0,
		LineID:      0,
		Segment:     1,
		Direction:   1, // moving toward Square, away from Circle
		Capacity:    6,
		Active:      true,
		JustArrived: true,
	})
	sim8.Step(0.01) // triggers boardAndAlight

	// Passenger should NOT board train heading in wrong direction (+1)
	check(len(sim8.State.Trains[0].Passengers) == 0, "passenger did not board train moving away from destination (-1 required, train moving +1)")
	check(len(sim8.State.Stations[1].Queue) == 1, "passenger remains in station queue")

	// Train on Line 0 at Station 1 heading -1 (toward Circle)
	sim8.State.Trains = append(sim8.State.Trains, engine.Train{
		ID:          1,
		LineID:      0,
		Segment:     1,
		Direction:   -1, // moving toward Circle
		Capacity:    6,
		Active:      true,
		JustArrived: true,
	})
	sim8.Step(0.01) // triggers boardAndAlight

	check(len(sim8.State.Trains[1].Passengers) == 1, "passenger boarded train moving in correct direction (-1)")
	check(len(sim8.State.Stations[1].Queue) == 0, "station queue emptied after boarding")

	// Test Transfer Alighting
	sim9 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	sim9.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	sim9.ApplyAction(engine.AddLine{Stations: []int{1, 2}})
	sim9.Step(0.01)

	// Train 0 on Line 0 carries passenger wanting Square (Station 2)
	sim9.State.Trains = append(sim9.State.Trains, engine.Train{
		ID:          0,
		LineID:      0,
		Segment:     1, // arriving at transfer Station 1
		Direction:   1,
		Capacity:    6,
		Active:      true,
		JustArrived: true,
		Passengers: []engine.Passenger{
			{Origin: 0, Destination: engine.Square},
		},
	})
	sim9.Step(0.01) // boardAndAlight triggers at Station 1

	check(len(sim9.State.Trains[0].Passengers) == 0, "passenger alighted from Line 0 train at transfer station")
	check(len(sim9.State.Stations[1].Queue) == 1, "alighted passenger joined transfer station queue")

	// Train 1 on Line 1 arrives at Station 1
	sim9.State.Trains = append(sim9.State.Trains, engine.Train{
		ID:          1,
		LineID:      1,
		Segment:     0, // Station 1 on Line 1 (1->2)
		Direction:   1,
		Capacity:    6,
		Active:      true,
		JustArrived: true,
	})
	sim9.Step(0.01)

	check(len(sim9.State.Trains[1].Passengers) == 1, "transferring passenger boarded Line 1 train")
	check(len(sim9.State.Stations[1].Queue) == 0, "transfer station queue cleared")

	// --- 11. Accelerating Passenger Spawn Rate ---
	header("11. Accelerating Passenger Spawn Rate")
	sim10 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
	})
	initialRate := sim10.CurrentSpawnRate()
	check(initialRate == 0.2, fmt.Sprintf("initial spawn rate = %.2f (want 0.20)", initialRate))

	// Step 1000 ticks
	for i := 0; i < 1000; i++ {
		if len(sim10.State.PendingRewardChoices) > 0 {
			sim10.ApplyAction(engine.ChooseReward{Choice: sim10.State.PendingRewardChoices[0]})
		}
		sim10.Step(0.01)
	}
	rateAfter1000 := sim10.CurrentSpawnRate()
	check(rateAfter1000 > initialRate, fmt.Sprintf("spawn rate accelerated from %.2f to %.2f after 1000 ticks", initialRate, rateAfter1000))

	// Step another 2000 ticks (total 3000 ticks)
	for i := 0; i < 2000; i++ {
		if len(sim10.State.PendingRewardChoices) > 0 {
			sim10.ApplyAction(engine.ChooseReward{Choice: sim10.State.PendingRewardChoices[0]})
		}
		sim10.Step(0.01)
	}
	rateAfter3000 := sim10.CurrentSpawnRate()
	check(rateAfter3000 > rateAfter1000, fmt.Sprintf("spawn rate accelerated further from %.2f to %.2f after 3000 ticks", rateAfter1000, rateAfter3000))

	// --- 12. Train Repositioning & Max Trains Per Line Limit ---
	header("12. Train Repositioning & Max Trains Per Line Limit")
	sim11 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
	})
	sim11.State.MaxTrainsPerLine = 2 // configurable per map limit
	sim11.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2}})

	// Add 2 trains to Line 0
	err = sim11.ApplyAction(engine.AddTrain{LineID: 0})
	check(err == nil, "1st train added to Line 0")
	err = sim11.ApplyAction(engine.AddTrain{LineID: 0})
	check(err == nil, "2nd train added to Line 0")

	// Try adding 3rd train to Line 0 (MaxTrainsPerLine is 2)
	err = sim11.ApplyAction(engine.AddTrain{LineID: 0})
	check(err != nil, "3rd train rejected because line reached MaxTrainsPerLine limit (2)")

	// Test RepositionTrain
	check(sim11.State.Trains[0].Segment == 0, "train 0 initially at segment 0")
	err = sim11.ApplyAction(engine.RepositionTrain{
		TrainID:   0,
		Segment:   2,
		Direction: -1,
	})
	check(err == nil, "RepositionTrain executed successfully")
	check(sim11.State.Trains[0].Segment == 2, "train 0 moved to segment 2")
	check(sim11.State.Trains[0].Direction == -1, "train 0 direction updated to -1")

	// --- 13. River Geometry & Water Crossings ---
	header("13. River Geometry & Water Crossings (Thick Rivers & 2D Water Polygons)")
	sim12 := engine.NewSimulatorWithWater(
		[]engine.Station{
			{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 10, Y: 20}},   // South of river
			{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 80}}, // North of river
			{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 40, Y: 80}},   // North of river
			{ID: 3, Kind: engine.Star, Pos: engine.Pos{X: 10, Y: 47}},     // Sits inside thick river channel (Y=50, width=10)
			{ID: 4, Kind: engine.Pentagon, Pos: engine.Pos{X: 70, Y: 70}}, // Inside 2D Water Polygon (50..90, 50..90)
		},
		[]engine.RiverSegment{
			{From: engine.Pos{X: 0, Y: 50}, To: engine.Pos{X: 100, Y: 50}, Width: 10.0},
		},
		[]engine.WaterPolygon{
			{Vertices: []engine.Pos{
				{X: 50, Y: 50}, {X: 90, Y: 50}, {X: 90, Y: 90}, {X: 50, Y: 90},
			}},
		},
	)

	// Segment 0->1 (10,20 -> 10,80) crosses Y=50 river; initial tunnels = 0
	err = sim12.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	check(err != nil, "AddLine across river rejected when pool has 0 tunnels")

	// Grant 1 tunnel token
	sim12.State.Resources.Grant(engine.RewardTunnel)
	err = sim12.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	check(err == nil, "AddLine across river succeeds after granting tunnel token")
	check(sim12.State.Resources.Tunnels == 0, "tunnel token spent on water crossing segment")
	check(sim12.State.Lines[0].TunnelAt[0], "line segment marked as tunnel crossing")

	// Non-river crossing segment 1->2 (10,80 -> 40,80; both North of river at Y=80)
	err = sim12.ApplyAction(engine.ExtendLine{LineID: 0, StationID: 2, UseTunnel: false})
	check(err == nil, "ExtendLine not crossing river succeeds without tunnel token")

	// Thick river channel test: Station 3 is at (10, 47), which is within Width=10/2 of center line Y=50
	err = sim12.ApplyAction(engine.ExtendLine{LineID: 0, StationID: 3, UseTunnel: false})
	check(err != nil, "ExtendLine into thick river channel requires tunnel token")

	// 2D Water Polygon test: Station 4 is inside water polygon at (70, 70)
	err = sim12.ApplyAction(engine.ExtendLine{LineID: 0, StationID: 4, UseTunnel: false})
	check(err != nil, "ExtendLine into 2D water polygon requires tunnel token")

	// Remove line 0 and verify tunnel token is refunded
	sim12.ApplyAction(engine.RemoveLine{LineID: 0})
	check(sim12.State.Resources.Tunnels == 1, "tunnel token refunded upon line removal")

	// --- 14. Expanded Station Kinds (Gem, Sector, Cross, Drop, Oval) ---
	header("14. Expanded Station Kinds (Rare Shapes)")
	sim13 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Gem, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Sector, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Cross, Pos: engine.Pos{X: 30, Y: 0}},
		{ID: 4, Kind: engine.Drop, Pos: engine.Pos{X: 40, Y: 0}},
		{ID: 5, Kind: engine.Oval, Pos: engine.Pos{X: 50, Y: 0}},
	})
	check(sim13.State.Stations[1].Kind.String() == "Gem", "Gem station kind initialized")
	check(sim13.State.Stations[2].Kind.String() == "Sector", "Sector station kind initialized")
	check(sim13.State.Stations[3].Kind.String() == "Cross", "Cross station kind initialized")
	check(sim13.State.Stations[4].Kind.String() == "Drop", "Drop station kind initialized")
	check(sim13.State.Stations[5].Kind.String() == "Oval", "Oval station kind initialized")

	sim13.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2, 3, 4, 5}})
	sim13.Step(0.01)

	rGem := engine.FindRoute(&sim13.State.Graph, &sim13.State, 0, engine.Gem)
	check(rGem.Reachable, "route from Circle to Gem station reachable")
	rOval := engine.FindRoute(&sim13.State.Graph, &sim13.State, 0, engine.Oval)
	check(rOval.Reachable, "route from Circle to Oval station reachable")

	// --- 15. Weighted Passenger Destination Sampling ---
	header("15. Weighted Passenger Destination Sampling (Rare Station Magnet Effect)")
	sim14 := engine.NewSimulator([]engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Star, Pos: engine.Pos{X: 20, Y: 0}},
	})
	rareDestinationSpawned := false
	activeFilteredOnly := true
	for i := 0; i < 500; i++ {
		sim14.Step(0.01)
		for _, st := range sim14.State.Stations {
			for _, p := range st.Queue {
				if p.Destination == engine.Star {
					rareDestinationSpawned = true
				}
				// Verify passenger destination matches kind of an alive station on the map
				foundActiveKind := false
				for _, s := range sim14.State.Stations {
					if s.Alive && s.Kind == p.Destination {
						foundActiveKind = true
						break
					}
				}
				if !foundActiveKind {
					activeFilteredOnly = false
				}
			}
		}
	}
	check(rareDestinationSpawned, "passengers spawned targeting rare active station (Star)")
	check(activeFilteredOnly, "passengers spawned only for active station kinds present on the map")

	// --- 16. City Map Presets ---
	header("16. City Map Presets (London, NYC, Tokyo)")
	simLondon := engine.NewSimulatorWithMap(engine.LondonMap())
	check(simLondon.State.MapName == "London", "London map initialized with name 'London'")
	check(len(simLondon.State.Rivers) == 3, "London map loaded River Thames geometry")
	check(simLondon.State.Resources.Tunnels == 1, "London map initialized with 1 tunnel token")

	simNYC := engine.NewSimulatorWithMap(engine.NYCMap())
	check(simNYC.State.MapName == "New York City", "NYC map initialized with name 'New York City'")
	check(len(simNYC.State.Rivers) == 2, "NYC map loaded Hudson and East Rivers")
	check(len(simNYC.State.WaterPolygons) == 1, "NYC map loaded Upper New York Bay polygon")
	check(simNYC.State.Resources.Tunnels == 2, "NYC map initialized with 2 tunnel tokens")

	simTokyo := engine.NewSimulatorWithMap(engine.TokyoMap())
	check(simTokyo.State.MapName == "Tokyo", "Tokyo map initialized with name 'Tokyo'")
	check(len(simTokyo.State.Rivers) == 1, "Tokyo map loaded Sumida River geometry")
	check(len(simTokyo.State.WaterPolygons) == 1, "Tokyo map loaded Tokyo Bay polygon")

	// Test water crossing on London Thames: station 0 (20,25) to station 1 (50,60) crosses Thames
	err = simLondon.ApplyAction(engine.AddLine{Stations: []int{0, 1}})
	check(err == nil, "AddLine across Thames River in London succeeds using 1 initial tunnel token")
	check(simLondon.State.Resources.Tunnels == 0, "London tunnel token spent on Thames River crossing")

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
