import time
from env import MiniMetroEnv

env = MiniMetroEnv()
env.reset()

# Warmup
env.step(0)

# Benchmark step
start = time.time()
for _ in range(100):
    env.step(0)
step_time = time.time() - start

print(f"Time for 100 steps: {step_time:.4f}s")
print(f"Time per step: {step_time / 100:.6f}s")

# Benchmark get_obs explicitly
start = time.time()
for _ in range(100):
    env._get_obs()
obs_time = time.time() - start

print(f"Time for 100 _get_obs: {obs_time:.4f}s")
print(f"Time per _get_obs: {obs_time / 100:.6f}s")
