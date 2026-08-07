package engine

type Action interface {
	isAction()
}

type AddLine struct {
	Stations []int
}

func (AddLine) isAction() {}

type ExtendLine struct {
	LineID    int
	StationID int
}

func (ExtendLine) isAction() {}

type AddTrain struct {
	LineID int
}

func (AddTrain) isAction() {}

type RemoveLine struct {
	LineID int
}

func (RemoveLine) isAction() {}
