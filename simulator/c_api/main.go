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

//export GetObservation
func GetObservation(handle C.uintptr_t, outNodes *C.float, outEdges *C.int32_t, outGlobals *C.float) {
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

	var edgeAttrsBuf []float32 // unused directly by python if not provided
	var globalsBuf []float32
	if outGlobals != nil {
		globalsBuf = unsafe.Slice((*float32)(unsafe.Pointer(outGlobals)), engine.GlobalFeatureDim)
	}

	sim.WriteVectorizedObservation(nodesBuf, edgesBuf, edgeAttrsBuf, globalsBuf)
}

//export GetActionMask
func GetActionMask(handle C.uintptr_t, outMask *C.uint8_t) {
	sim := getSim(uintptr(handle))
	if sim == nil || outMask == nil {
		return
	}

	maskSize := engine.MaxActionSpaceSize()
	maskSlice := unsafe.Slice((*uint8)(unsafe.Pointer(outMask)), maskSize)
	boolMask := make([]bool, maskSize)
	sim.GetActionMask(boolMask)

	for i := 0; i < maskSize; i++ {
		if boolMask[i] {
			maskSlice[i] = 1
		} else {
			maskSlice[i] = 0
		}
	}
}

func main() {}
