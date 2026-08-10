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

// ShortenLine removes one station from a line endpoint. FromFront=true removes the first station; false removes the last.
type ShortenLine struct {
	LineID    int
	FromFront bool
}

func (ShortenLine) isAction() {}
