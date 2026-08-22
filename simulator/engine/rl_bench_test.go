package engine

import (
	"testing"
)

func BenchmarkSimulatorStep(b *testing.B) {
	sim := NewSimulatorWithMap(LondonMap(), 12345)

	// Add a line so trains move passengers
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sim.Step(0.1)
	}
}

func BenchmarkSimulatorStepMacro(b *testing.B) {
	sim := NewSimulatorWithMap(LondonMap(), 12345)
	_ = sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _, done, _ := sim.StepMacro(nil, 1.0)
		if done {
			sim = NewSimulatorWithMap(LondonMap(), uint64(i))
			_ = sim.ApplyAction(AddLine{Stations: []int{0, 1, 2}})
		}
	}
}
