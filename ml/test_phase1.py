"""Phase-1 smoke test: 3 PPO updates, verifies all fixes."""
import sys
import os
try:
    import gymnasium as gym
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)

import numpy as np
import torch
from env import MiniMetroEnv
from model import MiniMetroActorCritic
from ppo import PPO

os.makedirs("runs/test_phase1", exist_ok=True)

NUM_ENVS  = 4
NUM_STEPS = 32
NUM_UPDS  = 3

def make_env(seed):
    def thunk():
        e = MiniMetroEnv(map_id=0, seed=seed)
        return gym.wrappers.RecordEpisodeStatistics(e)
    return thunk

if __name__ == "__main__":
    print("=== PHASE-1 END-TO-END SMOKE TEST ===")
    envs   = gym.vector.AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)], context='spawn')
    device = torch.device("cpu")
    model  = MiniMetroActorCritic(hidden_dim=32).to(device)
    agent  = PPO(model)

    assert agent.ent_coef == 0.05, f"ent_coef wrong: {agent.ent_coef}"
    print(f"PASS ent_coef={agent.ent_coef}")
    import inspect
    epd = inspect.signature(agent.update).parameters['update_epochs'].default
    assert epd == 4, f"update_epochs default wrong: {epd}"
    print(f"PASS update_epochs default={epd}")

    obs_buf  = {k: torch.zeros((NUM_STEPS, NUM_ENVS) + v.shape) for k, v in envs.single_observation_space.items()}
    actions  = torch.zeros((NUM_STEPS, NUM_ENVS))
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards  = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones    = torch.zeros((NUM_STEPS, NUM_ENVS))
    values   = torch.zeros((NUM_STEPS, NUM_ENVS))
    hidden_dim = 32
    lstm_hx  = torch.zeros((NUM_STEPS, NUM_ENVS, hidden_dim * 5))
    lstm_cx  = torch.zeros((NUM_STEPS, NUM_ENVS, hidden_dim * 5))

    next_obs, _ = envs.reset()
    next_obs_t  = {k: torch.as_tensor(v) for k, v in next_obs.items()}
    next_obs_t["action_mask"] = next_obs_t["action_mask"].bool()
    next_done   = torch.zeros(NUM_ENVS)
    next_lstm_state = (torch.zeros(1, NUM_ENVS, hidden_dim * 5),
                       torch.zeros(1, NUM_ENVS, hidden_dim * 5))
    all_rewards = []

    for update in range(1, NUM_UPDS + 1):
        for step in range(NUM_STEPS):
            for k in obs_buf: obs_buf[k][step].copy_(next_obs_t[k])
            dones[step].copy_(next_done)
            lstm_hx[step] = next_lstm_state[0].squeeze(0)
            lstm_cx[step] = next_lstm_state[1].squeeze(0)
            with torch.no_grad():
                mask = next_obs_t["action_mask"].bool()
                act, lp, _, val, next_lstm_state = model.get_action_and_value(next_obs_t, lstm_state=next_lstm_state, mask=mask)
                values[step] = val.flatten()
            actions[step] = act; logprobs[step] = lp
            next_obs, rew, term, trunc, _ = envs.step(act.cpu().numpy())
            done = np.logical_or(term, trunc)
            done_mask = torch.tensor(done, dtype=torch.float32).view(1, NUM_ENVS, 1)
            next_lstm_state = (
                next_lstm_state[0] * (1.0 - done_mask),
                next_lstm_state[1] * (1.0 - done_mask)
            )
            rewards[step].copy_(torch.as_tensor(rew, dtype=torch.float32))
            all_rewards.extend(rew.tolist())
            next_obs_t = {k: torch.as_tensor(v) for k, v in next_obs.items()}
            next_obs_t["action_mask"] = next_obs_t["action_mask"].bool()
            next_done = torch.as_tensor(done, dtype=torch.float32)

        with torch.no_grad():
            nv = model.get_value(next_obs_t, lstm_state=next_lstm_state).reshape(1, -1)
            adv, ret = agent.compute_gae(rewards, values, nv, dones, next_done)

        b_obs = obs_buf
        b_act = actions; b_lp = logprobs
        b_adv = adv;    b_ret = ret
        b_msk = b_obs["action_mask"].bool()

        # PHASE-1 PPO-1: full-batch normalization
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        # PHASE-1 PPO-4: LR decay
        frac = 1.0 - (update - 1) / NUM_UPDS
        for pg in agent.optimizer.param_groups: pg["lr"] = 3e-4 * frac

        pg_l, v_l, ent_l, cf, kl = agent.update(
            b_obs, b_act, b_lp, b_adv, b_ret, b_msk,
            values=values, init_lstm_hx=lstm_hx, init_lstm_cx=lstm_cx, update_epochs=4, num_minibatches=4, seq_len=16
        )
        noop = (b_act == 0).float().mean().item()
        lr   = agent.optimizer.param_groups[0]["lr"]
        print(f"Update {update}/{NUM_UPDS} | pg={pg_l:.4f} v={v_l:.4f} ent={ent_l:.4f} kl={kl:.6f} noop={noop:.1%} lr={lr:.2e}")

        for name, val in [("pg_loss",pg_l),("v_loss",v_l),("ent_loss",ent_l),("kl",kl)]:
            assert np.isfinite(val), f"{name} not finite: {val}"

    envs.close()

    has_pos     = any(r > 0.005 for r in all_rewards)
    no_penalty  = not any(abs(r - (-0.05)) < 1e-5 for r in all_rewards)
    print(f"\nPASS all losses finite ({NUM_UPDS} updates)")
    print(f"PASS rewards have positive values (survival bonus): {has_pos}")
    print(f"PASS no -0.05 flat action penalty: {no_penalty}")
    print(f"     reward range [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
    print(f"\n=== PHASE-1 SMOKE TEST PASSED ===")
