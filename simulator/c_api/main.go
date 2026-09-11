package main

/*
#include <stdint.h>
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"unsafe"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

var (
	registryMu sync.RWMutex
	registry   = make(map[uintptr]*engine.Simulator)
	nextHandle uintptr = 1
)

func registerSim(sim *engine.Simulator) uintptr {
	registryMu.Lock()
	defer registryMu.Unlock()
	h := nextHandle
	nextHandle++
	registry[h] = sim
	return h
}

func getSim(handle uintptr) *engine.Simulator {
	registryMu.RLock()
	defer registryMu.RUnlock()
	return registry[handle]
}

func unregisterSim(handle uintptr) {
	registryMu.Lock()
	defer registryMu.Unlock()
	delete(registry, handle)
}

//export CreateSimulator
func CreateSimulator(mapID C.int, seed C.uint64_t) C.uintptr_t {
	var cfg engine.MapConfig
	switch mapID {
	case 1:
		cfg = engine.NYCMap()
	case 2:
		cfg = engine.TokyoMap()
	default:
		cfg = engine.LondonMap()
	}

	sim := engine.NewSimulatorWithMap(cfg, uint64(seed))
	h := registerSim(sim)
	return C.uintptr_t(h)
}

//export FreeSimulator
func FreeSimulator(handle C.uintptr_t) {
	unregisterSim(uintptr(handle))
}

//export Step
func Step(handle C.uintptr_t, actionID C.int, duration C.float, outReward *C.float, outDone *C.uint8_t) {
	sim := getSim(uintptr(handle))
	if sim == nil {
		if outDone != nil {
			*outDone = 1
		}
		return
	}

	action, _ := engine.ActionFromIndex(int(actionID))
	_, reward, done, _ := sim.StepMacro(action, float64(duration))

	if outReward != nil {
		*outReward = C.float(reward)
	}
	if outDone != nil {
		if done {
			*outDone = 1
		} else {
			*outDone = 0
		}
	}
}

//export StepWithBreakdown
func StepWithBreakdown(handle C.uintptr_t, actionID C.int, duration C.float, outReward *C.float, outDone *C.uint8_t, outBreakdown *C.float) {
	sim := getSim(uintptr(handle))
	if sim == nil {
		if outDone != nil {
			*outDone = 1
		}
		return
	}

	action, _ := engine.ActionFromIndex(int(actionID))
	_, reward, done, _, rb := sim.StepMacroBreakdown(action, float64(duration))

	if outReward != nil {
		*outReward = C.float(reward)
	}
	if outDone != nil {
		if done {
			*outDone = 1
		} else {
			*outDone = 0
		}
	}
	if outBreakdown != nil {
		slice := unsafe.Slice((*float32)(unsafe.Pointer(outBreakdown)), 7)
		slice[0] = float32(rb.Delivery)
		slice[1] = float32(rb.Connectivity)
		slice[2] = float32(rb.CrowdPenalty)
		slice[3] = float32(rb.GameOver)
		slice[4] = float32(rb.Redundancy)
		slice[5] = float32(rb.LoopReversal)
		slice[6] = float32(rb.TrackEfficiency)
	}
}

//export SetScoringConfig
func SetScoringConfig(handle C.uintptr_t, alphaCrowd C.float, betaGameOver C.float, connectivityBonus C.float, trackEfficiency C.float, linearCrowd C.uint8_t) {
	sim := getSim(uintptr(handle))
	if sim == nil {
		return
	}
	sim.ScoringConfig = engine.ScoringConfig{
		AlphaCrowdPenalty:     float64(alphaCrowd),
		BetaGameOverPenalty:   float64(betaGameOver),
		ConnectivityBonus:     float64(connectivityBonus),
		TrackEfficiencyWeight: float64(trackEfficiency),
		LinearCrowdPenalty:    linearCrowd != 0,
		Initialized:           true,
	}
}

//export GetTotalTrackLength
func GetTotalTrackLength(handle C.uintptr_t) C.float {
	sim := getSim(uintptr(handle))
	if sim == nil {
		return 0.0
	}
	return C.float(sim.TotalTrackLength())
}

//export GetObservation
func GetObservation(handle C.uintptr_t, outNodes *C.float, outEdges *C.int32_t, outEdgeAttrs *C.float, outGlobals *C.float, outNumNodes *C.int32_t, outNumEdges *C.int32_t) {
	sim := getSim(uintptr(handle))
	if sim == nil {
		return
	}

	const maxNodes = 30
	const maxEdges = 200

	var nodesBuf []float32
	if outNodes != nil {
		nodesBuf = unsafe.Slice((*float32)(unsafe.Pointer(outNodes)), maxNodes*engine.NodeFeatureDim)
	}

	var edgesBuf []int32
	if outEdges != nil {
		edgesBuf = unsafe.Slice((*int32)(unsafe.Pointer(outEdges)), maxEdges*2)
	}

	var edgeAttrsBuf []float32
	if outEdgeAttrs != nil {
		edgeAttrsBuf = unsafe.Slice((*float32)(unsafe.Pointer(outEdgeAttrs)), maxEdges*engine.EdgeFeatureDim)
	}

	var globalsBuf []float32
	if outGlobals != nil {
		globalsBuf = unsafe.Slice((*float32)(unsafe.Pointer(outGlobals)), engine.GlobalFeatureDim)
	}

	numNodes, numEdges := sim.WriteVectorizedObservation(nodesBuf, edgesBuf, edgeAttrsBuf, globalsBuf)

	// PHASE-2: return exact counts so Python can avoid fragile heuristic detection.
	if outNumNodes != nil {
		*outNumNodes = C.int32_t(numNodes)
	}
	if outNumEdges != nil {
		*outNumEdges = C.int32_t(numEdges)
	}
}

var (
	boolMaskBuf []bool
	boolMaskMu  sync.Mutex
)

//export GetActionMask
func GetActionMask(handle C.uintptr_t, outMask *C.uint8_t) {
	sim := getSim(uintptr(handle))
	if sim == nil || outMask == nil {
		return
	}

	maskSize := engine.MaxActionSpaceSize()
	maskSlice := unsafe.Slice((*uint8)(unsafe.Pointer(outMask)), maskSize)

	boolMaskMu.Lock()
	if len(boolMaskBuf) < maskSize {
		boolMaskBuf = make([]bool, maskSize)
	}
	boolMask := boolMaskBuf[:maskSize]
	sim.GetActionMask(boolMask)

	for i := 0; i < maskSize; i++ {
		if boolMask[i] {
			maskSlice[i] = 1
		} else {
			maskSlice[i] = 0
		}
	}
	boolMaskMu.Unlock()
}

//export SetPendingReward
func SetPendingReward(handle C.uintptr_t, c0 C.int, c1 C.int) {
	sim := getSim(uintptr(handle))
	if sim == nil {
		return
	}
	if c0 < 0 {
		sim.State.PendingRewardChoices = nil
	} else if c1 < 0 {
		sim.State.PendingRewardChoices = []engine.RewardType{engine.RewardType(c0)}
	} else {
		sim.State.PendingRewardChoices = []engine.RewardType{engine.RewardType(c0), engine.RewardType(c1)}
	}
}

//export ReverseLine
func ReverseLine(handle C.uintptr_t, lineID C.int) C.int {
	sim := getSim(uintptr(handle))
	if sim == nil {
		return -1
	}
	if err := sim.ReverseLine(int(lineID)); err != nil {
		return -1
	}
	return 0
}

func main() {}
