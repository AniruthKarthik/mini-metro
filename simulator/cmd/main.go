package main

import (
	"fmt"
	"strings"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

func stationKindName(k engine.StationKind) string {
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

func main() {
	fmt.Println("==================================================")
	fmt.Println("      MINI METRO SIMULATOR DEMO RUNNER            ")
	fmt.Println("==================================================")

	// 1. Create Initial Stations
	stations := []engine.Station{
		{ID: 0, Kind: engine.Circle, Pos: engine.Pos{X: 0, Y: 0}},
		{ID: 1, Kind: engine.Triangle, Pos: engine.Pos{X: 10, Y: 0}},
		{ID: 2, Kind: engine.Square, Pos: engine.Pos{X: 20, Y: 0}},
		{ID: 3, Kind: engine.Star, Pos: engine.Pos{X: 30, Y: 0}},
		{ID: 4, Kind: engine.Pentagon, Pos: engine.Pos{X: 40, Y: 0}},
	}

	sim := engine.NewSimulator(stations)

	fmt.Printf("Initialized Simulator with %d stations.\n\n", len(sim.State.Stations))

	// 2. Add Metro Line 0: Stations 0 -> 1 -> 2 -> 3 -> 4
	fmt.Println("[Action] Adding Line 0 (Stations: 0 -> 1 -> 2 -> 3 -> 4)...")
	sim.ApplyAction(engine.AddLine{Stations: []int{0, 1, 2, 3, 4}})

	// 3. Add Train on Line 0
	fmt.Println("[Action] Deploying Train 0 on Line 0...")
	sim.ApplyAction(engine.AddTrain{LineID: 0})

	fmt.Println("\nStarting Simulation Loop (50 steps with dt = 0.2)...")
	fmt.Println(strings.Repeat("-", 50))

	// 4. Run Simulation
	for step := 1; step <= 50; step++ {
		sim.Step(0.2)

		obs := sim.Observation()

		fmt.Printf("Tick: %-3d | Score: %-3d | Alive: %-5t | Trains Active: %d\n",
			obs.Tick, obs.Score, sim.State.Alive, len(obs.TrainLineIDs))

		// Print Station status every 5 ticks or if score changes
		if step%5 == 0 || !sim.State.Alive {
			fmt.Printf("   Stations Overview:\n")
			for _, st := range sim.State.Stations {
				fmt.Printf("     - Station %d (%-8s): Queue Len = %d\n",
					st.ID, stationKindName(st.Kind), len(st.Queue))
			}
			fmt.Printf("   Trains Overview:\n")
			for _, tr := range sim.State.Trains {
				if tr.Active {
					fmt.Printf("     - Train %d (Line %d): Segment %d, Direction %+d, Progress %.2f, Passengers = %d/%d\n",
						tr.ID, tr.LineID, tr.Segment, tr.Direction, tr.Progress, len(tr.Passengers), tr.Capacity)
				}
			}
			fmt.Println(strings.Repeat("-", 50))
		}

		if !sim.State.Alive {
			fmt.Println("\n[GAME OVER] Station queue limit exceeded!")
			break
		}
	}

	fmt.Println("\n==================================================")
	fmt.Printf("  FINAL SCORE: %d | TOTAL TICKS: %d\n", sim.State.Score, sim.State.Tick)
	fmt.Println("==================================================")
}
