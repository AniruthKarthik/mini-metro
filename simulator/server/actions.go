package server

import (
	"encoding/json"
	"fmt"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
)

// {"type":"add_line",        "payload":{"stations":[0,1,2]}}
// {"type":"extend_line",     "payload":{"line_id":0,"station_id":3,"use_tunnel":false}}
// {"type":"add_train",       "payload":{"line_id":0}}
// {"type":"remove_line",     "payload":{"line_id":1}}
// {"type":"choose_reward",   "payload":{"choice":2}}   // 0=Line 1=Train 2=Tunnel 3=Carriage 4=Interchange
// {"type":"add_carriage",    "payload":{"train_id":0}}
// {"type":"upgrade_interchange","payload":{"station_id":2}}
// {"type":"shorten_line",    "payload":{"line_id":0,"from_front":false}}
// {"type":"close_loop",      "payload":{"line_id":0,"use_tunnel":false}}
// {"type":"open_loop",       "payload":{"line_id":0}}
// {"type":"reposition_train","payload":{"train_id":0,"segment":2,"direction":1}}
// {"type":"set_speed",       "payload":{"tps":60}}   // server-side control
// {"type":"pause"}
// {"type":"resume"}
// {"type":"restart"}
type ActionEnvelope struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

func ParseAction(raw []byte) (engine.Action, string, error) {
	var env ActionEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, "", fmt.Errorf("invalid JSON: %w", err)
	}

	switch env.Type {
	// ── server-side controls ─────────────────────────────────────
	case "pause", "resume", "restart":
		return nil, env.Type, nil

	case "set_speed":
		return nil, env.Type + ":" + string(env.Payload), nil

	// ── engine actions ────────────────────────────────────────────
	case "add_line":
		var p struct {
			Stations []int `json:"stations"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.AddLine{Stations: p.Stations}, "", nil

	case "extend_line":
		var p struct {
			LineID    int  `json:"line_id"`
			StationID int  `json:"station_id"`
			UseTunnel bool `json:"use_tunnel"`
			FromFront bool `json:"from_front"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.ExtendLine{LineID: p.LineID, StationID: p.StationID, UseTunnel: p.UseTunnel, FromFront: p.FromFront}, "", nil

	case "insert_station":
		var p struct {
			LineID    int  `json:"line_id"`
			StationID int  `json:"station_id"`
			Index     int  `json:"index"`
			UseTunnel bool `json:"use_tunnel"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.InsertStation{LineID: p.LineID, StationID: p.StationID, Index: p.Index, UseTunnel: p.UseTunnel}, "", nil

	case "add_train":
		var p struct {
			LineID int `json:"line_id"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.AddTrain{LineID: p.LineID}, "", nil

	case "remove_line":
		var p struct {
			LineID int `json:"line_id"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.RemoveLine{LineID: p.LineID}, "", nil

	case "choose_reward":
		var p struct {
			Choice int `json:"choice"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.ChooseReward{Choice: engine.RewardType(p.Choice)}, "", nil

	case "add_carriage":
		var p struct {
			TrainID int `json:"train_id"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.AddCarriage{TrainID: p.TrainID}, "", nil

	case "upgrade_interchange":
		var p struct {
			StationID int `json:"station_id"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.UpgradeInterchange{StationID: p.StationID}, "", nil

	case "shorten_line":
		var p struct {
			LineID    int  `json:"line_id"`
			FromFront bool `json:"from_front"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.ShortenLine{LineID: p.LineID, FromFront: p.FromFront}, "", nil

	case "close_loop":
		var p struct {
			LineID    int  `json:"line_id"`
			UseTunnel bool `json:"use_tunnel"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.CloseLoop{LineID: p.LineID, UseTunnel: p.UseTunnel}, "", nil

	case "open_loop":
		var p struct {
			LineID int `json:"line_id"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.OpenLoop{LineID: p.LineID}, "", nil

	case "reposition_train":
		var p struct {
			TrainID   int `json:"train_id"`
			Segment   int `json:"segment"`
			Direction int `json:"direction"`
		}
		if err := json.Unmarshal(env.Payload, &p); err != nil {
			return nil, "", err
		}
		return engine.RepositionTrain{
			TrainID:   p.TrainID,
			Segment:   p.Segment,
			Direction: p.Direction,
		}, "", nil

	default:
		return nil, "", fmt.Errorf("unknown action type: %q", env.Type)
	}
}
