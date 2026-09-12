import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.995, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5):  # ent_coef=0.05 (PPO-3), gamma=0.995 (PPO-6)
        self.model = model
        use_cuda = torch.cuda.is_available() and next(self.model.parameters()).is_cuda
        if use_cuda:
            try:
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5, fused=True)
            except Exception:
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def compute_gae(self, rewards, values, next_value, dones, next_done):
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        num_steps = rewards.shape[0]
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, logprobs, advantages, returns, masks, values=None, init_lstm_hx=None, init_lstm_cx=None, update_epochs=4, num_minibatches=4, seq_len=16, target_kl=0.015):
        T, B = actions.shape
        num_chunks = T // seq_len
        b_size = B * num_chunks
        minibatch_size = b_size // num_minibatches
        
        inds = torch.arange(b_size)
        clipfracs = []
        
        device = next(self.model.parameters()).device
        use_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        
        # Reshape inputs to [num_chunks, B, seq_len, ...] and then flatten chunk and B dims
        # So shape becomes [b_size, seq_len, ...]
        def reshape_to_chunks(t):
            chunked = t.view(num_chunks, seq_len, B, *t.shape[2:])
            return chunked.transpose(1, 2).reshape(b_size, seq_len, *t.shape[2:])
            
        b_obs = {k: reshape_to_chunks(v) for k, v in obs.items()}
        b_actions = reshape_to_chunks(actions)
        b_logprobs = reshape_to_chunks(logprobs)
        b_advantages = reshape_to_chunks(advantages)
        b_returns = reshape_to_chunks(returns)
        b_masks = reshape_to_chunks(masks)
        b_values = reshape_to_chunks(values) if values is not None else None
            
        # Extract initial LSTM states for each chunk. Shape: [T, B, H] -> [num_chunks, seq_len, B, H]
        b_hx = init_lstm_hx.view(num_chunks, seq_len, B, -1)[:, 0, :, :].reshape(b_size, -1)
        b_cx = init_lstm_cx.view(num_chunks, seq_len, B, -1)[:, 0, :, :].reshape(b_size, -1)
        
        stop_early = False
        for epoch in range(update_epochs):
            if stop_early:
                break
            torch.randperm(b_size, out=inds)
            for start in range(0, b_size, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                
                mb_obs = {k: v[mbinds].to(device, non_blocking=True) for k, v in b_obs.items()}
                mb_actions = b_actions[mbinds].to(device, non_blocking=True)
                mb_logprobs = b_logprobs[mbinds].to(device, non_blocking=True)
                mb_advantages = b_advantages[mbinds].to(device, non_blocking=True)
                mb_returns = b_returns[mbinds].to(device, non_blocking=True)
                mb_masks = b_masks[mbinds].to(device, non_blocking=True)
                mb_values = b_values[mbinds].to(device, non_blocking=True) if b_values is not None else None

                mb_hx = b_hx[mbinds].to(device, non_blocking=True).unsqueeze(0)  # [1, minibatch_size, H]
                mb_cx = b_cx[mbinds].to(device, non_blocking=True).unsqueeze(0)
                mb_lstm_state = (mb_hx, mb_cx)
                
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    # For action and value, get_action_and_value handles [B, T, ...] correctly
                    _, newlogprob, entropy, newvalue, _ = self.raw_model.get_action_and_value(
                        mb_obs, lstm_state=mb_lstm_state, action=mb_actions, mask=mb_masks
                    )
                    
                    # newlogprob is [minibatch_size, seq_len]
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    
                    if target_kl is not None and approx_kl.item() > target_kl:
                        stop_early = True
                        break

                    mb_adv = mb_advantages
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    newvalue = newvalue.squeeze(-1) # [minibatch_size, seq_len]
                    if mb_values is not None:
                        v_clipped = mb_values + torch.clamp(
                            newvalue - mb_values, -self.clip_coef, self.clip_coef
                        )
                        v_loss1 = (newvalue - mb_returns) ** 2
                        v_loss2 = (v_clipped - mb_returns) ** 2
                        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
        return pg_loss.item(), v_loss.item(), entropy_loss.item(), np.mean(clipfracs), approx_kl.item()
