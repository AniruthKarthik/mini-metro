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

type InsertStation struct {
	LineID    int
	StationID int
	Index     int // index in line.Stations where StationID will be inserted; clamped to [1, len-1]
	// Tunnel tokens are managed automatically: net crossings for the two new segments
	// minus the replaced crossing are spent/refunded without explicit caller input.
}

func (InsertStation) isAction() {}

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

type RemoveCarriage struct {
	TrainID int
}

func (RemoveCarriage) isAction() {}

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
	LineID    int // target line ID (if < 0, keeps current line)
	Segment   int
	Direction int
}

func (RepositionTrain) isAction() {}
