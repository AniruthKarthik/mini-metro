import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

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

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_masks, update_epochs=4, num_minibatches=4):
        b_size = b_actions.shape[0]
        minibatch_size = b_size // num_minibatches
        
        inds = torch.arange(b_size)
        clipfracs = []
        
        for epoch in range(update_epochs):
            torch.randperm(b_size, out=inds)
            for start in range(0, b_size, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                
                mb_obs = {k: v[mbinds] for k, v in b_obs.items()}
                
                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(mb_obs, b_actions[mbinds], mask=b_masks[mbinds])
                logratio = newlogprob - b_logprobs[mbinds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mbinds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mbinds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
        return pg_loss.item(), v_loss.item(), entropy_loss.item(), np.mean(clipfracs), approx_kl.item()
