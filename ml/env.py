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

class MiniMetroEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Mini Metro Go simulator.
    """
    def __init__(self, map_id=-1, seed=None):
        super().__init__()
        
        self.map_id = map_id
        if seed is None:
            self._seed_val = np.random.randint(0, 2**31)
        else:
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
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed_val = seed
        else:
            self._seed_val = np.random.randint(0, 2**31)

        if self.handle is not None:
            lib.FreeSimulator(self.handle)
            
        # Curriculum Learning: Random Map Selection
        current_map = self.map_id
        if current_map == -1:
            current_map = int(np.random.choice([0, 1, 2])) # London, NYC, Tokyo
            
        self.handle = lib.CreateSimulator(current_map, self._seed_val)
        
        obs = self._get_obs()
        info = {}
        return obs, info
        
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
            
            # Dynamic Frame Skipping: tick up to 4 times (4 seconds total)
            for step_idx in range(4):
                curr_action = action_id if step_idx == 0 else 0
                
                lib.Step(self.handle, curr_action, duration, ctypes.byref(self._out_reward), ctypes.byref(self._out_done))
                
                step_done = bool(self._out_done.value)
                step_reward = float(self._out_reward.value)
                
                if not step_done:
                    step_reward += 0.01  # Survival bonus
                    
                obs = self._get_obs()
                num_stations = int(obs["num_nodes"][0])
                emergency = False
                
                for i in range(num_stations):
                    node = obs["nodes"][i]
                    overcrowd_progress = float(node[22])
                    if overcrowd_progress > 0:
                        step_reward -= 0.3 * overcrowd_progress
                        emergency = True
                        
                    raw_queue_total = float(node[12:22].sum())
                    fill_approx = raw_queue_total / 6.0
                    if fill_approx > 0.8:
                        step_reward -= 0.1 * (fill_approx - 0.8)
                        
                total_reward += step_reward
                if step_done:
                    done = True
                    break
                    
                if emergency:
                    break

            return obs, total_reward, done, False, info
        except Exception:
            obs, info = self.reset()
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

if __name__ == "__main__":
    env = MiniMetroEnv()
    obs, info = env.reset()
    print("Observation globals:", obs["globals"])
    print("Num valid actions:", obs["action_mask"].sum())
    
    # Try random valid action
    valid_actions = np.where(obs["action_mask"])[0]
    action = np.random.choice(valid_actions)
    
    obs, reward, done, _, _ = env.step(action)
    print("Step reward:", reward, "Done:", done)
    env.close()
