"""PPO-Clip with GAE, normalized physical inputs and independent value/policy nets."""
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from rocket import Rocket


class PhysicalMLP(nn.Module):
    def __init__(self, output_dim, target_y):
        super().__init__()
        self.register_buffer('target_y', torch.tensor(float(target_y)))
        self.layers = nn.Sequential(nn.Linear(9, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, output_dim))
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.layers[-1].weight, gain=.01 if output_dim > 1 else 1.)

    def forward(self, state):
        s = state.reshape(-1, 8) * 100  # Rocket.flatten scales every field by 100.
        x,y,vx,vy,theta,omega,t,phi = s.unbind(-1)
        features = torch.stack((x/300, (y-self.target_y)/300, vx/50, vy/50,
                                theta.sin(), theta.cos(), omega, t/800, phi/(np.pi/9)), -1)
        return self.layers(features)


class PPOActorCritic(nn.Module):
    def __init__(self, task='hover', learning_rate=3e-4):
        super().__init__()
        target_y = 200 if task == 'hover' else 25
        self.actor = PhysicalMLP(9, target_y)
        self.critic = PhysicalMLP(1, target_y)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def forward(self, states):
        return self.softmax(self.actor(states)), self.critic(states)

    def get_action(self, state, deterministic=False, exploration=0.):
        state = torch.as_tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        probs, value = self(state)
        distribution = Categorical(probs=probs)
        action = probs.argmax(-1) if deterministic else distribution.sample()
        return action.item(), distribution.log_prob(action)[0], value[0]


def calculate_gae(rewards, values, terminated, truncated, timeout_values, last_values, gamma=.99, gae_lambda=.95):
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(last_values)
    for t in reversed(range(len(rewards))):
        next_value = last_values if t == len(rewards)-1 else values[t+1]
        next_value = torch.where(truncated[t], timeout_values[t], next_value)
        next_value = torch.where(terminated[t], torch.zeros_like(next_value), next_value)
        delta = rewards[t] + gamma * next_value - values[t]
        continuation = ~(terminated[t] | truncated[t])
        running = delta + gamma * gae_lambda * continuation * running
        advantages[t] = running
    return advantages, advantages + values


class PPOCollector:
    def __init__(self, task, num_envs, max_steps, render_every=0, landing_reward="legacy",
                 drag_model="quadratic", drag_coefficient=.004, hover_reward="legacy"):
        self.envs = [Rocket(max_steps, task, render_enabled=(i == 0 and render_every > 0), landing_reward=landing_reward,
                            drag_model=drag_model, drag_coefficient=drag_coefficient, hover_reward=hover_reward)
                     for i in range(num_envs)]
        self.states = np.stack([env.reset() for env in self.envs])
        self.episode_rewards = np.zeros(num_envs)
        self.completed = 0
        self.steps = 0
        self.render_every = render_every
        self.preview_episodes = 0
        self.hover_windows = [deque(maxlen=100) for _ in self.envs]

    @torch.no_grad()
    def collect(self, net, horizon, gamma=.99, gae_lambda=.95):
        n = len(self.envs)
        observations = np.empty((horizon, n, 8), dtype=np.float32)
        actions = torch.empty((horizon, n), dtype=torch.long)
        rewards = torch.empty((horizon, n))
        terminated = torch.zeros((horizon, n), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        timeout_values = torch.zeros((horizon, n))
        finished_rewards = []
        self.finished_successes = []
        old_values = torch.empty((horizon, n))
        old_log_probs = torch.empty((horizon, n))
        for t in range(horizon):
            observations[t] = self.states
            probs, value = net(torch.from_numpy(self.states))
            distribution = Categorical(probs=probs)
            actions[t] = distribution.sample()
            old_log_probs[t] = distribution.log_prob(actions[t])
            old_values[t] = value.squeeze(-1)
            next_states, timeouts, timeout_states = [], [], []
            for i, (env, action) in enumerate(zip(self.envs, actions[t].tolist())):
                state, reward, done, _ = env.step(action)
                timeout = env.step_id >= env.max_steps and not done
                rewards[t, i] = reward
                deadline_failure = timeout and env.task == 'landing' and env.landing_reward == 'shaped_v1'
                terminated[t, i] = done or deadline_failure
                truncated[t, i] = timeout and not deadline_failure
                self.episode_rewards[i] += reward
                if env.task == "hover":
                    self.hover_windows[i].append(env.is_stable_hover(env.state))
                if i == 0 and self.render_every and self.preview_episodes % self.render_every == 0:
                    env.render(window_name='Rocket recycling - training', wait_time=1)
                if timeout and not deadline_failure:
                    timeouts.append(i)
                    timeout_states.append(state)
                if done or timeout:
                    finished_rewards.append(float(self.episode_rewards[i]))
                    success = (env.already_landing if env.task == 'landing' else
                               (timeout and len(self.hover_windows[i]) == 100 and sum(self.hover_windows[i]) >= 80))
                    self.finished_successes.append(bool(success))
                    self.hover_windows[i].clear()
                    self.episode_rewards[i] = 0
                    self.completed += 1
                    if i == 0:
                        self.preview_episodes += 1
                    state = env.reset()
                next_states.append(state)
            if timeouts:
                timeout_values[t, timeouts] = net.critic(torch.from_numpy(np.stack(timeout_states))).squeeze(-1)
            self.states = np.stack(next_states)
        last_values = net.critic(torch.from_numpy(self.states)).squeeze(-1)
        advantages, returns = calculate_gae(rewards, old_values, terminated, truncated, timeout_values, last_values, gamma, gae_lambda)
        self.steps += horizon * n
        return (torch.from_numpy(observations.reshape(-1, 8)), actions.reshape(-1),
                old_log_probs.reshape(-1), advantages.reshape(-1), returns.reshape(-1), finished_rewards)


def update_ppo(net, batch, epochs=4, minibatch_size=256, clip_ratio=.2,
               entropy_coef=.01, value_coef=.5, target_kl=.02):
    states, actions, old_log_probs, advantages, returns, _ = batch
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    last_actor = last_critic = 0.
    for epoch in range(epochs):
        # Measure KL over the full batch before starting another reuse epoch.
        with torch.no_grad():
            probs, _ = net(states)
            log_ratio = Categorical(probs=probs).log_prob(actions) - old_log_probs
            kl = ((log_ratio.exp()-1)-log_ratio).mean().item()
        if kl > target_kl:
            break
        for indices in torch.randperm(len(states)).split(minibatch_size):
            probs, values = net(states[indices])
            distribution = Categorical(probs=probs)
            ratio = (distribution.log_prob(actions[indices])-old_log_probs[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1-clip_ratio,1+clip_ratio) * advantages[indices]
            actor_loss = -torch.minimum(unclipped,clipped).mean()
            critic_loss = .5*(returns[indices]-values.squeeze(-1)).square().mean()
            loss = actor_loss + value_coef*critic_loss - entropy_coef*distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            net.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), .5, error_if_nonfinite=True)
            net.optimizer.step()
            last_actor, last_critic = actor_loss.item(), critic_loss.item()
    return last_actor, last_critic
