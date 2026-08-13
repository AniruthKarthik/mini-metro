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
	UseTunnel bool // set true to spend a tunnel token for a long-distance segment
	FromFront bool // set true to extend from the first station rather than the last
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

type ChooseReward struct{ Choice RewardType }

func (ChooseReward) isAction() {}

type AddCarriage struct {
	TrainID int
}

func (AddCarriage) isAction() {}

type UpgradeInterchange struct {
	StationID int
}

func (UpgradeInterchange) isAction() {}

type ShortenLine struct {
	LineID    int
	FromFront bool
}

func (ShortenLine) isAction() {}

type CloseLoop struct {
	LineID    int
	UseTunnel bool
}

func (CloseLoop) isAction() {}

type OpenLoop struct {
	LineID int
}

func (OpenLoop) isAction() {}

type RepositionTrain struct {
	TrainID   int
	Segment   int
	Direction int
}

func (RepositionTrain) isAction() {}
