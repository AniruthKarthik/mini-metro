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

type ChooseReward struct{ Choice RewardType }

func (ChooseReward) isAction() {}

type AddCarriage struct {
	TrainID int
}

func (AddCarriage) isAction() {}

// UpgradeInterchange spends one interchange token to mark a station as an interchange hub.
type UpgradeInterchange struct {
	StationID int
}

func (UpgradeInterchange) isAction() {}
