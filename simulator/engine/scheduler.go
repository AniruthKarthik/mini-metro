package engine

// EventKind identifies what type of simulation event fires.
type EventKind int

const (
	EventReward       EventKind = iota // weekly reward offer
	EventSpawnStation                  // dynamic station spawn
)

// ScheduledEvent is a single time-based trigger with a tick and kind.
type ScheduledEvent struct {
	FireAt uint64
	Kind   EventKind
}

// EventScheduler holds pending events ordered by FireAt (ascending).
type EventScheduler struct {
	Events []ScheduledEvent
}

// Schedule inserts a new event into the scheduler, maintaining FireAt order.
func (es *EventScheduler) Schedule(firAt uint64, kind EventKind) {
	ev := ScheduledEvent{FireAt: firAt, Kind: kind}
	i := 0
	for i < len(es.Events) && es.Events[i].FireAt <= firAt {
		i++
	}
	es.Events = append(es.Events, ScheduledEvent{})
	copy(es.Events[i+1:], es.Events[i:])
	es.Events[i] = ev
}

// Poll removes and returns all events whose FireAt <= currentTick.
func (es *EventScheduler) Poll(currentTick uint64) []ScheduledEvent {
	i := 0
	for i < len(es.Events) && es.Events[i].FireAt <= currentTick {
		i++
	}
	due := append([]ScheduledEvent(nil), es.Events[:i]...)
	es.Events = es.Events[i:]
	return due
}
