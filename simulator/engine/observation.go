package engine

type Observation struct {
	StationKinds []StationKind
	StationQueues []int
	TrainLineIDs []int
	TrainSegments []int
	TrainLoads []int
	Score int
	Tick uint64
}

func (s *Simulator) Observation() Observation {
	obs := Observation{}

	for _, st := range s.State.Stations {
		obs.StationKinds = append(obs.StationKinds, st.Kind)
		obs.StationQueues = append(obs.StationQueues, len(st.Queue))
	}

	for _, tr := range s.State.Trains {
		if !tr.Active {
			continue
		}

		obs.TrainLineIDs = append(obs.TrainLineIDs, tr.LineID)
		obs.TrainSegments = append(obs.TrainSegments, tr.Segment)
		obs.TrainLoads = append(obs.TrainLoads, len(tr.Passengers))
	}

	obs.Score = s.State.Score
	obs.Tick = s.State.Tick

	return obs
}