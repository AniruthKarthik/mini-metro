import ctypes
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Load the Go shared library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libminimetro.so")
try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
except OSError as e:
    print(f"Warning: Could not load {lib_path}. Run build_lib.sh first. {e}")
    lib = None

if lib:
    # extern uintptr_t CreateSimulator(int mapID, uint64_t seed);
    lib.CreateSimulator.argtypes = [ctypes.c_int, ctypes.c_uint64]
    lib.CreateSimulator.restype = ctypes.c_void_p

    # extern void FreeSimulator(uintptr_t handle);
    lib.FreeSimulator.argtypes = [ctypes.c_void_p]

    # extern void Step(uintptr_t handle, int actionID, float duration, float* outReward, uint8_t* outDone);
    lib.Step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)]

    # extern void GetObservation(uintptr_t handle, float* outNodes, int32_t* outEdges,
    #                             float* outEdgeAttrs, float* outGlobals,
    #                             int32_t* outNumNodes, int32_t* outNumEdges);
    lib.GetObservation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),  # PHASE-2: outNumNodes
        ctypes.POINTER(ctypes.c_int32),  # PHASE-2: outNumEdges
    ]

    # extern void GetActionMask(uintptr_t handle, uint8_t* outMask);
    lib.GetActionMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]

    # extern void SetPendingReward(uintptr_t handle, int c0, int c1);
    lib.SetPendingReward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    # extern void StepWithBreakdown(uintptr_t handle, int actionID, float duration, float* outReward, uint8_t* outDone, float* outBreakdown);
    if hasattr(lib, "StepWithBreakdown"):
        lib.StepWithBreakdown.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]

    # extern void SetScoringConfig(uintptr_t handle, float alphaCrowd, float betaGameOver, float connectivityBonus, float trackEfficiency, uint8_t linearCrowd);
    if hasattr(lib, "SetScoringConfig"):
        lib.SetScoringConfig.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint8,
        ]

    # extern float GetTotalTrackLength(uintptr_t handle);
    if hasattr(lib, "GetTotalTrackLength"):
        lib.GetTotalTrackLength.argtypes = [ctypes.c_void_p]
        lib.GetTotalTrackLength.restype = ctypes.c_float

    # extern int ReverseLine(uintptr_t handle, int lineID);
    if hasattr(lib, "ReverseLine"):
        lib.ReverseLine.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.ReverseLine.restype = ctypes.c_int

class MiniMetroEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Mini Metro Go simulator.
    """
    def __init__(self, map_id=-1, seed=None):
        super().__init__()
        
        self.map_id = map_id
        self._seed_val = seed
            
        self.handle = None
        
        # Dimensions from observation.go and action_space.go
        self.max_nodes = 30
        self.max_edges = 200  # From c_api/main.go maxEdges
        self.node_dim = 32    # PHASE-5: was 29; +incoming_train_count, incoming_train_load, nearest_train_proximity
        self.edge_dim = 10
        self.global_dim = 23  # was 13; +two 5-dim one-hot reward card encodings (indices 13..22)
        self.action_space_size = 4087  # PHASE-4: was 4108; AddCarriage reduced 28→7 (now lineID-indexed)

        # Observation space definition
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, self.node_dim), dtype=np.float32),
            "edges": spaces.Box(low=0, high=self.max_nodes, shape=(2, self.max_edges), dtype=np.int32),
            "edge_attrs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_edges, self.edge_dim), dtype=np.float32),
            "globals": spaces.Box(low=-np.inf, high=np.inf, shape=(self.global_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.bool_),
            "num_nodes": spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int32),
            "num_edges": spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=np.int32)
        })

        self.action_space = spaces.Discrete(self.action_space_size)

        # Pre-allocate ctypes buffers
        self._out_nodes = (ctypes.c_float * (self.max_nodes * self.node_dim))()
        self._out_edges = (ctypes.c_int32 * (self.max_edges * 2))()
        self._out_edge_attrs = (ctypes.c_float * (self.max_edges * self.edge_dim))()
        self._out_globals = (ctypes.c_float * self.global_dim)()
        
        self._out_reward = ctypes.c_float(0.0)
        self._out_done = ctypes.c_uint8(0)
        self._out_mask = (ctypes.c_uint8 * self.action_space_size)()
        # PHASE-2: exact node/edge counts from C API (replaces fragile heuristic)
        self._out_num_nodes = ctypes.c_int32(0)
        self._out_num_edges = ctypes.c_int32(0)
        # P2-1, P2-2: decomposed reward breakdown buffer (7 channels from Go)
        self._out_breakdown = (ctypes.c_float * 7)()
        self.scoring_config = {
            "alpha_crowd": 0.30,
            "beta_game_over": 200.0,
            "connectivity_bonus": 2.0,
            "track_efficiency": 0.01,
            "linear_crowd": False,
        }
        self._episode_reward_breakdown = {
            "delivery": 0.0,
            "survival": 0.0,
            "connectivity": 0.0,
            "crowd_penalty": 0.0,
            "game_over": 0.0,
            "redundancy": 0.0,
            "loop_reversal": 0.0,
            "track_efficiency": 0.0,
        }
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed_val = seed
        elif self._seed_val is None:
            self._seed_val = np.random.randint(0, 2**31)

        if self.handle is not None:
            lib.FreeSimulator(self.handle)
            
        # Curriculum Learning: Random Map Selection
        current_map = self.map_id
        if current_map == -1:
            current_map = int(np.random.choice([0, 1, 2])) # London, NYC, Tokyo
            
        self.handle = lib.CreateSimulator(current_map, self._seed_val)
        self._apply_scoring_config()
        self._episode_reward_breakdown = {
            "delivery": 0.0,
            "survival": 0.0,
            "connectivity": 0.0,
            "crowd_penalty": 0.0,
            "game_over": 0.0,
            "redundancy": 0.0,
            "loop_reversal": 0.0,
            "track_efficiency": 0.0,
        }
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def set_scoring_config(self, alpha_crowd=0.30, beta_game_over=200.0, connectivity_bonus=2.0, track_efficiency=0.01, linear_crowd=False):
        """Configure or ablate reward coefficients at runtime (P2-1, P2-2)."""
        self.scoring_config = {
            "alpha_crowd": float(alpha_crowd),
            "beta_game_over": float(beta_game_over),
            "connectivity_bonus": float(connectivity_bonus),
            "track_efficiency": float(track_efficiency),
            "linear_crowd": bool(linear_crowd),
        }
        self._apply_scoring_config()

    def _apply_scoring_config(self):
        if lib is not None and hasattr(lib, "SetScoringConfig") and self.handle is not None:
            lib.SetScoringConfig(
                self.handle,
                ctypes.c_float(self.scoring_config["alpha_crowd"]),
                ctypes.c_float(self.scoring_config["beta_game_over"]),
                ctypes.c_float(self.scoring_config["connectivity_bonus"]),
                ctypes.c_float(self.scoring_config["track_efficiency"]),
                ctypes.c_uint8(1 if self.scoring_config["linear_crowd"] else 0),
            )

    def get_total_track_length(self) -> float:
        """Query the current total rail track length across all lines (P2-2)."""
        if self.handle is not None and lib is not None and hasattr(lib, "GetTotalTrackLength"):
            return float(lib.GetTotalTrackLength(self.handle))
        return 0.0

    def reverse_line(self, line_id: int) -> bool:
        """
        Reverse the station sequence of an active non-loop line in the simulator (P2-3).
        Preserves exact physical network geometry while swapping front (0) and back (1) endpoints.
        """
        if self.handle is not None and lib is not None and hasattr(lib, "ReverseLine"):
            res = lib.ReverseLine(self.handle, int(line_id))
            return res == 0
        return False

    def randomize_line_orientations(self, p: float = 0.5):
        """
        Rollout data augmentation (P2-3):
        Randomly reverses active non-loop lines with probability p.
        Eliminates positional/array-index bias between front (0) and tail (1) endpoints.
        """
        if self.handle is None:
            return
        for line_id in range(7):
            if np.random.rand() < p:
                self.reverse_line(line_id)

    def _get_obs(self):
        try:
            # Zero out node/edge buffers (globals are always fully written).
            ctypes.memset(self._out_nodes, 0, ctypes.sizeof(self._out_nodes))
            ctypes.memset(self._out_edges, 0, ctypes.sizeof(self._out_edges))
            ctypes.memset(self._out_edge_attrs, 0, ctypes.sizeof(self._out_edge_attrs))

            lib.GetObservation(
                self.handle,
                self._out_nodes,
                self._out_edges,
                self._out_edge_attrs,
                self._out_globals,
                ctypes.byref(self._out_num_nodes),
                ctypes.byref(self._out_num_edges),
            )

            lib.GetActionMask(self.handle, self._out_mask)

            nodes       = np.ctypeslib.as_array(self._out_nodes).reshape(self.max_nodes, self.node_dim).copy()
            edges       = np.ctypeslib.as_array(self._out_edges).reshape(self.max_edges, 2).transpose().copy()
            edge_attrs  = np.ctypeslib.as_array(self._out_edge_attrs).reshape(self.max_edges, self.edge_dim).copy()
            globals_feat= np.ctypeslib.as_array(self._out_globals).copy()
            action_mask = np.ctypeslib.as_array(self._out_mask).astype(bool).copy()

            # Ensure action mask has at least one valid action (fallback to NoOp action 0)
            if not action_mask.any():
                action_mask[0] = True

            num_nodes = max(1, int(self._out_num_nodes.value))
            num_edges = max(0, int(self._out_num_edges.value))

            return {
                "nodes":       nodes,
                "edges":       edges,
                "edge_attrs":  edge_attrs,
                "globals":     globals_feat,
                "action_mask": action_mask,
                "num_nodes":   np.array([num_nodes], dtype=np.int32),
                "num_edges":   np.array([num_edges], dtype=np.int32),
            }
        except Exception:
            fallback_mask = np.zeros(self.action_space_size, dtype=bool)
            fallback_mask[0] = True
            return {
                "nodes":       np.zeros((self.max_nodes, self.node_dim), dtype=np.float32),
                "edges":       np.zeros((2, self.max_edges), dtype=np.int32),
                "edge_attrs":  np.zeros((self.max_edges, self.edge_dim), dtype=np.float32),
                "globals":     np.zeros((self.global_dim,), dtype=np.float32),
                "action_mask": fallback_mask,
                "num_nodes":   np.array([1], dtype=np.int32),
                "num_edges":   np.array([0], dtype=np.int32),
            }

    def step(self, action):
        if self.handle is None:
            obs, info = self.reset()
            return obs, 0.0, True, False, info
            
        try:
            action_id = int(action)
            duration = 1.0 
            
            total_reward = 0.0
            done = False
            info = {}
            
            step_breakdown = {
                "delivery": 0.0,
                "survival": 0.0,
                "connectivity": 0.0,
                "crowd_penalty": 0.0,
                "game_over": 0.0,
                "redundancy": 0.0,
                "loop_reversal": 0.0,
                "track_efficiency": 0.0,
            }

            # Dynamic Frame Skipping: tick up to 4 times (4 seconds total)
            for step_idx in range(4):
                curr_action = action_id if step_idx == 0 else 0
                
                if hasattr(lib, "StepWithBreakdown"):
                    lib.StepWithBreakdown(
                        self.handle,
                        curr_action,
                        duration,
                        ctypes.byref(self._out_reward),
                        ctypes.byref(self._out_done),
                        self._out_breakdown,
                    )
                    go_delivery = float(self._out_breakdown[0])
                    go_connectivity = float(self._out_breakdown[1])
                    go_crowd = float(self._out_breakdown[2])
                    go_game_over = float(self._out_breakdown[3])
                    go_redundancy = float(self._out_breakdown[4])
                    go_loop_reversal = float(self._out_breakdown[5])
                    go_track_efficiency = float(self._out_breakdown[6])
                else:
                    lib.Step(self.handle, curr_action, duration, ctypes.byref(self._out_reward), ctypes.byref(self._out_done))
                    go_delivery = float(self._out_reward.value)
                    go_connectivity = 0.0
                    go_crowd = 0.0
                    go_game_over = 0.0
                    go_redundancy = 0.0
                    go_loop_reversal = 0.0
                    go_track_efficiency = 0.0

                step_done = bool(self._out_done.value)
                sub_survival = 0.01 if not step_done else 0.0
                
                obs = self._get_obs()
                num_stations = int(obs["num_nodes"][0])
                emergency = False
                sub_py_crowd = 0.0
                
                for i in range(num_stations):
                    node = obs["nodes"][i]
                    overcrowd_progress = float(node[22])
                    if overcrowd_progress > 0:
                        sub_py_crowd -= 0.3 * overcrowd_progress
                        emergency = True
                        
                    raw_queue_total = float(node[12:22].sum())
                    fill_approx = raw_queue_total / 6.0
                    if fill_approx > 0.8:
                        sub_py_crowd -= 0.1 * (fill_approx - 0.8)
                        
                step_breakdown["delivery"] += go_delivery
                step_breakdown["connectivity"] += go_connectivity
                step_breakdown["crowd_penalty"] += (go_crowd + sub_py_crowd)
                step_breakdown["game_over"] += go_game_over
                step_breakdown["redundancy"] += go_redundancy
                step_breakdown["loop_reversal"] += go_loop_reversal
                step_breakdown["track_efficiency"] += go_track_efficiency
                step_breakdown["survival"] += sub_survival

                sub_step_reward = (
                    go_delivery
                    + go_connectivity
                    + go_crowd
                    + sub_py_crowd
                    + go_game_over
                    + go_redundancy
                    + go_loop_reversal
                    + go_track_efficiency
                    + sub_survival
                )
                total_reward += sub_step_reward

                if step_done:
                    done = True
                    break
                    
                if emergency:
                    break

            for k, v in step_breakdown.items():
                self._episode_reward_breakdown[k] += v

            info["reward_breakdown"] = step_breakdown
            info["episode_reward_breakdown"] = dict(self._episode_reward_breakdown)
            info["total_track_length"] = self.get_total_track_length()

            return obs, total_reward, done, False, info
        except Exception:
            obs, info = self.reset()
            info["reward_breakdown"] = {k: 0.0 for k in self._episode_reward_breakdown}
            info["episode_reward_breakdown"] = dict(self._episode_reward_breakdown)
            info["total_track_length"] = 0.0
            return obs, 0.0, True, False, info
        
    def set_pending_reward(self, card0, card1):
        """Mock or set pending reward choices (for testing/probing). Use -1 to clear."""
        if self.handle is not None and lib is not None and hasattr(lib, "SetPendingReward"):
            lib.SetPendingReward(self.handle, int(card0), int(card1))
            return self._get_obs()
        return None

    def close(self):
        if self.handle is not None and lib is not None:
            try:
                lib.FreeSimulator(self.handle)
            except Exception:
                pass
            self.handle = None

class LineOrientationAugmentation(gym.Wrapper):
    """
    Gym wrapper for P2-3: Line Endpoint Symmetry Augmentation.
    Randomly reverses active non-loop line orientations with probability flip_prob
    before returning observations, eliminating array-index bias during training.
    """
    def __init__(self, env: gym.Env, flip_prob: float = 0.5):
        super().__init__(env)
        self.flip_prob = flip_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated) and self.flip_prob > 0:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "randomize_line_orientations"):
                unwrapped.randomize_line_orientations(self.flip_prob)
                obs = unwrapped._get_obs()
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    env = MiniMetroEnv()
    obs, info = env.reset()
    print("Observation globals:", obs["globals"])
    print("Num valid actions:", obs["action_mask"].sum())
    
    # Try random valid action
    valid_actions = np.where(obs["action_mask"])[0]
    action = np.random.choice(valid_actions)
    
    obs, reward, done, _, info = env.step(action)
    print("Step reward:", reward, "Done:", done)
    print("Step breakdown:", info.get("reward_breakdown"))
    print("Episode breakdown:", info.get("episode_reward_breakdown"))
    env.close()
