package engine

import (
	"fmt"
	"testing"
)

// TestWeeklyRewardSamplingAudit1000 runs a 1000-event empirical probe of weekly rewards.
func TestWeeklyRewardSamplingAudit1000(t *testing.T) {
	sim := NewSimulatorWithMap(LondonMap(), 42)

	rewardNames := map[RewardType]string{
		RewardLine:        "LINE",
		RewardTrain:       "LOCOMOTIVE",
		RewardTunnel:      "TUNNEL",
		RewardCarriage:    "CARRIAGE",
		RewardInterchange: "INTERCHANGE",
	}

	counts := make(map[string]int)
	posCounts := [2]map[string]int{make(map[string]int), make(map[string]int)}
	pairCounts := make(map[string]int)
	linesOfferedEvents := 0
	linesSelectedEvents := 0
	linesRejectedDueToLimit := 0

	type EventRecord struct {
		Week        int
		Option1     string
		Option2     string
		Selected    string
		LinesBefore int
		LinesAfter  int
		ChoiceIdx   int
	}
	var sampleTable []EventRecord

	totalEvents := 1000

	for week := 1; week <= totalEvents; week++ {
		linesBefore := sim.State.Resources.Lines
		activeLines := 0
		for _, l := range sim.State.Lines {
			if !l.Removed {
				activeLines++
			}
		}
		unlockedBefore := activeLines + linesBefore

		// Trigger offerReward
		sim.offerReward()

		if len(sim.State.PendingRewardChoices) != 2 {
			t.Fatalf("Week %d: expected 2 pending reward choices, got %d", week, len(sim.State.PendingRewardChoices))
		}

		c0 := sim.State.PendingRewardChoices[0]
		c1 := sim.State.PendingRewardChoices[1]

		name0 := rewardNames[c0]
		name1 := rewardNames[c1]

		counts[name0]++
		counts[name1]++
		posCounts[0][name0]++
		posCounts[1][name1]++

		pairKey := fmt.Sprintf("%s + %s", name0, name1)
		pairCounts[pairKey]++

		hasLine := (c0 == RewardLine || c1 == RewardLine)
		if hasLine {
			linesOfferedEvents++
		}

		// Verify line limit enforcement: if unlockedBefore >= 7, LINE must NOT be offered
		if unlockedBefore >= 7 && hasLine {
			t.Fatalf("Week %d: LINE was offered despite unlockedLines=%d reaching maxLines=7", week, unlockedBefore)
		}
		if unlockedBefore >= 7 {
			linesRejectedDueToLimit++
		}

		// Select LINE if offered, otherwise pick c0
		choiceIdx := 0
		if c1 == RewardLine {
			choiceIdx = 1
		} else if c0 == RewardLine {
			choiceIdx = 0
		}

		selectedType := sim.State.PendingRewardChoices[choiceIdx]
		selectedName := rewardNames[selectedType]
		if selectedType == RewardLine {
			linesSelectedEvents++
		}

		// Apply action
		err := sim.ApplyAction(ChooseReward{Choice: RewardType(choiceIdx)})
		if err != nil {
			t.Fatalf("Week %d: failed to apply ChooseReward: %v", week, err)
		}

		linesAfter := sim.State.Resources.Lines

		if selectedType == RewardLine {
			if linesAfter != linesBefore+1 {
				t.Fatalf("Week %d: selected LINE, expected lines %d -> %d, got %d", week, linesBefore, linesBefore+1, linesAfter)
			}
		}

		if week <= 15 || week%100 == 0 {
			sampleTable = append(sampleTable, EventRecord{
				Week:        week,
				Option1:     name0,
				Option2:     name1,
				Selected:    selectedName,
				LinesBefore: linesBefore,
				LinesAfter:  linesAfter,
				ChoiceIdx:   choiceIdx,
			})
		}
	}

	fmt.Println("\n==========================================================================")
	fmt.Println("  WEEKLY UPGRADE FORENSIC AUDIT: 1000 CONTROLLED REWARD EVENTS")
	fmt.Println("==========================================================================")
	fmt.Printf("%-6s | %-12s | %-12s | %-12s | %-10s | %-10s | %-8s\n",
		"Week", "Option 1", "Option 2", "Selected", "Lines Bef", "Lines Aft", "ChoiceIdx")
	fmt.Println("--------------------------------------------------------------------------")
	for _, row := range sampleTable {
		fmt.Printf("%-6d | %-12s | %-12s | %-12s | %-10d | %-10d | %-8d\n",
			row.Week, row.Option1, row.Option2, row.Selected, row.LinesBefore, row.LinesAfter, row.ChoiceIdx)
	}

	fmt.Println("\n--- EMPIRICAL STATISTICAL SUMMARY (1000 EVENTS) ---")
	fmt.Printf("Total events: %d\n", totalEvents)
	fmt.Printf("Events offering LINE: %d (%.2f%%)\n", linesOfferedEvents, float64(linesOfferedEvents)/float64(totalEvents)*100.0)
	fmt.Printf("LINE at Position 0 (choice 0): %d\n", posCounts[0]["LINE"])
	fmt.Printf("LINE at Position 1 (choice 1): %d\n", posCounts[1]["LINE"])
	fmt.Printf("Events where LINE was selected: %d\n", linesSelectedEvents)
	fmt.Printf("Events where LINE was suppressed by line limit (7 reached): %d\n", linesRejectedDueToLimit)

	fmt.Println("\n--- CARD APPEARANCE COUNTS ---")
	for name, count := range counts {
		fmt.Printf("  - %-14s: %4d cards (%5.1f%% of all cards, %5.1f%% of events)\n",
			name, count, float64(count)/float64(totalEvents*2)*100.0, float64(count)/float64(totalEvents)*100.0)
	}
}

// TestLineLimitCondition verifies line availability across all states from 0 lines used to 7 lines used.
func TestLineLimitCondition(t *testing.T) {
	// London has 3 initial lines, max 7 lines.
	cfg := LondonMap()
	if cfg.MaxLines != 7 {
		t.Fatalf("expected MaxLines=7, got %d", cfg.MaxLines)
	}
	sim := NewSimulatorWithMap(cfg, 123)

	// Test with different numbers of lines used (created):
	for used := 0; used <= 7; used++ {
		sim.State.Lines = make([]Line, used)
		for i := 0; i < used; i++ {
			sim.State.Lines[i] = Line{ID: i, Stations: []int{0, 1}, Removed: false}
		}

		for unused := 0; unused <= 7-used+1; unused++ {
			sim.State.Resources.Lines = unused
			sim.State.PendingRewardChoices = nil

			sim.offerReward()

			hasLine := (sim.State.PendingRewardChoices[0] == RewardLine || sim.State.PendingRewardChoices[1] == RewardLine)
			unlocked := used + unused

			if unlocked >= 7 && hasLine {
				t.Fatalf("used=%d, unused=%d (total unlocked=%d >= 7): RewardLine should NOT be offered!", used, unused, unlocked)
			}
		}
	}
}

// TestRewardApplicationLifecycle tests full lifecycle:
// 1. Start with N available lines.
// 2. Force/generate weekly Line reward.
// 3. Apply reward.
// 4. Verify unlocked line count increases by exactly 1.
// 5. Create newly unlocked line with AddLine.
// 6. Verify simulator accepts it.
// 7. Verify line resource is consumed correctly.
// Also tests: line selected but never used, multiple New Lines, reaching max lines, followed by other upgrades.
func TestRewardApplicationLifecycle(t *testing.T) {
	stations := []Station{
		{ID: 0, Kind: Circle, Pos: Pos{X: 10, Y: 10}, Alive: true, Capacity: 6},
		{ID: 1, Kind: Triangle, Pos: Pos{X: 20, Y: 10}, Alive: true, Capacity: 6},
		{ID: 2, Kind: Square, Pos: Pos{X: 30, Y: 10}, Alive: true, Capacity: 6},
		{ID: 3, Kind: Star, Pos: Pos{X: 40, Y: 10}, Alive: true, Capacity: 6},
		{ID: 4, Kind: Pentagon, Pos: Pos{X: 50, Y: 10}, Alive: true, Capacity: 6},
	}
	sim := NewSimulator(stations, 42)
	sim.State.Resources.Lines = 1
	sim.State.Resources.Trains = 5
	sim.State.Lines = nil

	// Step 1: Start with 1 line available
	if sim.State.Resources.Lines != 1 {
		t.Fatalf("expected 1 line initially, got %d", sim.State.Resources.Lines)
	}

	// Step 2: Force weekly Line reward
	sim.State.PendingRewardChoices = []RewardType{RewardLine, RewardCarriage}

	// Step 3: Apply ChooseReward for choice 0 (Line)
	err := sim.ApplyAction(ChooseReward{Choice: 0})
	if err != nil {
		t.Fatalf("ChooseReward failed: %v", err)
	}

	// Step 4: Verify lines increased by exactly 1 (1 -> 2)
	if sim.State.Resources.Lines != 2 {
		t.Fatalf("expected 2 lines after reward, got %d", sim.State.Resources.Lines)
	}

	// Step 5 & 6: Create the newly unlocked line
	err = sim.ApplyAction(AddLine{Stations: []int{0, 1}})
	if err != nil {
		t.Fatalf("AddLine(0, 1) failed: %v", err)
	}

	// Step 7: Verify line resource was consumed (2 -> 1)
	if sim.State.Resources.Lines != 1 {
		t.Fatalf("expected 1 line remaining after AddLine, got %d", sim.State.Resources.Lines)
	}
	if len(sim.State.Lines) != 1 || sim.State.Lines[0].Removed {
		t.Fatalf("expected 1 active line in sim.State.Lines")
	}

	// Subtest: Line selected but never used (remains available in reserve)
	sim.State.PendingRewardChoices = []RewardType{RewardCarriage, RewardLine}
	err = sim.ApplyAction(ChooseReward{Choice: 1})
	if err != nil {
		t.Fatalf("ChooseReward choice 1 failed: %v", err)
	}
	if sim.State.Resources.Lines != 2 {
		t.Fatalf("expected 2 lines after second reward, got %d", sim.State.Resources.Lines)
	}

	// Subtest: Multiple New Line rewards up to max lines (7)
	for sim.State.Resources.Lines+len(sim.State.Lines) < 7 {
		sim.State.PendingRewardChoices = []RewardType{RewardLine, RewardTunnel}
		sim.ApplyAction(ChooseReward{Choice: 0})
	}
	totalUnlocked := sim.State.Resources.Lines + len(sim.State.Lines)
	if totalUnlocked != 7 {
		t.Fatalf("expected 7 total unlocked lines, got %d", totalUnlocked)
	}

	// Now offer reward: verify RewardLine is no longer generated
	sim.offerReward()
	for _, c := range sim.State.PendingRewardChoices {
		if c == RewardLine {
			t.Fatalf("RewardLine offered after 7 lines were unlocked!")
		}
	}

	// Subtest: AddLine followed by other upgrades (Carriage, Train)
	err = sim.ApplyAction(AddLine{Stations: []int{1, 2}})
	if err != nil {
		t.Fatalf("second AddLine failed: %v", err)
	}
	// Apply Carriage upgrade
	sim.State.Resources.Grant(RewardCarriage)
	err = sim.ApplyAction(AddCarriage{LineID: 0})
	if err != nil {
		t.Fatalf("AddCarriage failed: %v", err)
	}
}
