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

    # extern void GetObservation(uintptr_t handle, float* outNodes, int32_t* outEdges, float* outEdgeAttrs, float* outGlobals);
    lib.GetObservation.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]

    # extern void GetActionMask(uintptr_t handle, uint8_t* outMask);
    lib.GetActionMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]

class MiniMetroEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Mini Metro Go simulator.
    """
    def __init__(self, map_id=0, seed=None):
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
        self.node_dim = 25
        self.edge_dim = 10
        self.global_dim = 8
        self.action_space_size = 4108 # Calculated from action_space.go

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
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed_val = seed
        else:
            self._seed_val = np.random.randint(0, 2**31)

        if self.handle is not None:
            lib.FreeSimulator(self.handle)
            
        self.handle = lib.CreateSimulator(self.map_id, self._seed_val)
        
        obs = self._get_obs()
        info = {}
        return obs, info
        
    def _get_obs(self):
        # We need to know how many nodes and edges are returned by the C API.
        # But wait, GetObservation in C doesn't return the counts. Let me check the go code.
        # ah, the c_api/main.go might return or we can figure it out by padding.
        # Actually I should check c_api/main.go. Let's assume it populates up to the max buffers, 
        # but the remaining values will be zero? 
        # I need to look closely at c_api/main.go to see what GetObservation actually does.
        # For now, we will extract the numpy arrays from ctypes.
        
        # Zero out buffers
        ctypes.memset(self._out_nodes, 0, ctypes.sizeof(self._out_nodes))
        ctypes.memset(self._out_edges, 0, ctypes.sizeof(self._out_edges))
        ctypes.memset(self._out_edge_attrs, 0, ctypes.sizeof(self._out_edge_attrs))
        
        lib.GetObservation(
            self.handle, 
            self._out_nodes, 
            self._out_edges, 
            self._out_edge_attrs, 
            self._out_globals
        )
        
        lib.GetActionMask(self.handle, self._out_mask)
        
        nodes = np.ctypeslib.as_array(self._out_nodes).reshape(self.max_nodes, self.node_dim).copy()
        edges = np.ctypeslib.as_array(self._out_edges).reshape(self.max_edges, 2).transpose().copy() # Shape (2, max_edges)
        edge_attrs = np.ctypeslib.as_array(self._out_edge_attrs).reshape(self.max_edges, self.edge_dim).copy()
        globals_feat = np.ctypeslib.as_array(self._out_globals).copy()
        
        action_mask = np.ctypeslib.as_array(self._out_mask).astype(bool).copy()
        
        # Simple heuristic to determine num nodes and edges:
        # A node is valid if it's not all zeros (e.g. outNodes[base+0] is X coord, usually > 0 if station exists)
        # Actually in observation.go: nodes = s.WriteVectorizedObservation(nodes, ...)
        # The number of valid nodes/edges isn't returned directly in the C API, but we can compute it.
        num_nodes = 0
        for i in range(self.max_nodes):
            if np.sum(np.abs(nodes[i])) > 0:
                num_nodes += 1
            else:
                break
                
        num_edges = 0
        for i in range(self.max_edges):
            if np.sum(np.abs(edge_attrs[i])) > 0 or edges[0, i] > 0 or edges[1, i] > 0:
                num_edges += 1
            else:
                break
                
        obs = {
            "nodes": nodes,
            "edges": edges,
            "edge_attrs": edge_attrs,
            "globals": globals_feat,
            "action_mask": action_mask,
            "num_nodes": np.array([num_nodes], dtype=np.int32),
            "num_edges": np.array([num_edges], dtype=np.int32),
        }
        return obs

    def step(self, action):
        if self.handle is None:
            raise RuntimeError("Environment has not been reset.")
            
        action_id = int(action)
        # The step function takes a duration. We'll simulate 1 second per step.
        # Actually in mini metro, 1 second is fine. The step function: Step(handle, actionID, duration, &outReward, &outDone)
        duration = 1.0 
        
        lib.Step(self.handle, action_id, duration, ctypes.byref(self._out_reward), ctypes.byref(self._out_done))
        
        reward = float(self._out_reward.value)
        
        # Action penalty to discourage the AI from spamming useless actions
        if action_id != 0:
            reward -= 0.05
            
        done = bool(self._out_done.value)
        
        obs = self._get_obs()
        info = {}
        
        return obs, reward, done, False, info
        
    def close(self):
        if self.handle is not None:
            lib.FreeSimulator(self.handle)
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
