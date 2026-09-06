import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from ppo import PPOActorCritic, PPOCollector, update_ppo
from training_visuals import RewardPlot, preview


if __name__ == '__main__':

    # Edit these settings directly, then Run in PyCharm.
    task = 'hover'  # 'hover' or 'landing'
    max_m_episode = 800000  # total completed episodes, including resumed progress
    max_steps = 800
    num_envs = 32
    rollout_steps = 128
    seed = 0  # reproducible fresh training
    landing_reward = "shaped_v1"  # "legacy" for the original reward
    hover_reward = "legacy"  # fastest acquisition in 3-seed tests; shaped_v1 favors precision
    drag_model = "quadratic"
    drag_coefficient = 0.004  # m^-1; terminal speed ~49.5 m/s
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    ppo_epochs = 4
    minibatch_size = 256
    entropy_coef = 0.01

    show_curve = True
    plot_every = 10       # refresh the reward curve every 10 updates
    render_every = 1000   # preview one episode every 1000 completed training episodes; 0 disables
    save_every = 1000     # save curve and model every 1000 completed episodes
    resume = True

    os.chdir(Path(__file__).resolve().parent)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = {'drag_model': drag_model, 'drag_coefficient': drag_coefficient,
                   'landing_reward': landing_reward, 'hover_reward': hover_reward}
    ckpt_folder = Path(task + '_ppo_quadratic_v1_ckpt')
    ckpt_folder.mkdir(exist_ok=True)
    ckpt_path = ckpt_folder / 'ppo_latest.pt'

    last_episode_id = -1
    update_id = 0
    total_steps = 0
    REWARDS = []
    SUCCESSES = []
    reward_episode_start = 0
    net = PPOActorCritic(task=task, learning_rate=learning_rate)

    # PPO uses its own state encoding and checkpoint; old A2C files stay intact.
    source = ckpt_path
    if resume and source.exists():
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        if checkpoint.get('algorithm') != 'ppo':
            raise ValueError('Expected a PPO checkpoint')
        if checkpoint.get('task', task) != task:
            raise ValueError('Checkpoint task does not match task')
        if checkpoint.get('environment') != environment:
            raise ValueError('Checkpoint physics/reward differs; use a new checkpoint folder')
        if task == 'landing' and checkpoint.get('landing_reward', 'legacy') != landing_reward:
            raise ValueError('Checkpoint reward version does not match landing_reward')
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = list(checkpoint.get('REWARDS', []))
        SUCCESSES = list(checkpoint.get('SUCCESSES', []))
        reward_episode_start = checkpoint.get('reward_episode_start', last_episode_id + 1 - len(REWARDS))
        if checkpoint.get('format_version') == 3:
            net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            update_id = checkpoint.get('updates', 0)
            total_steps = checkpoint.get('steps', 0)
        for group in net.optimizer.param_groups:
            group['lr'] = learning_rate
        print('loaded:', source, 'completed episodes:', last_episode_id + 1)

    # Headless training environments; preview uses a separate environment.
    collector = PPOCollector(task, num_envs, max_steps, **environment)
    episode_base = last_episode_id + 1
    steps_base = total_steps
    completed_episodes = episode_base
    plot = RewardPlot(task, show=show_curve)
    plot.update(REWARDS, reward_episode_start)
    next_render = completed_episodes + 1  # first preview soon after starting/resuming
    next_save = (completed_episodes // save_every + 1) * save_every
    started = time.perf_counter()

    def save_progress():
        temporary = ckpt_path.with_suffix('.tmp')
        torch.save({'format_version': 3, 'algorithm': 'ppo', 'task': task, 'landing_reward': landing_reward,
                    'environment': environment,
                    'hyperparameters': {'seed': seed, 'gamma': gamma, 'gae_lambda': gae_lambda,
                                        'learning_rate': learning_rate, 'ppo_epochs': ppo_epochs,
                                        'minibatch_size': minibatch_size, 'entropy_coef': entropy_coef},
                    'episode_id': completed_episodes - 1,
                    'updates': update_id, 'steps': steps_base + collector.steps,
                    'REWARDS': REWARDS, 'SUCCESSES': SUCCESSES, 'reward_episode_start': reward_episode_start,
                    'model_G_state_dict': net.state_dict(),
                    'optimizer_state_dict': net.optimizer.state_dict()}, temporary)
        temporary.replace(ckpt_path)
        plot.update(REWARDS, reward_episode_start)
        plot.save(ckpt_folder / 'rewards_latest.jpg')

    try:
        while completed_episodes < max_m_episode:
            # Collect on-policy data, then reuse it with clipped PPO updates.
            batch = collector.collect(net, rollout_steps, gamma, gae_lambda)
            actor_loss, critic_loss = update_ppo(net, batch, epochs=ppo_epochs,
                                                minibatch_size=minibatch_size, entropy_coef=entropy_coef)
            rewards = batch[-1]
            REWARDS.extend(rewards)
            SUCCESSES.extend(collector.finished_successes)
            update_id += 1
            completed_episodes = episode_base + collector.completed

            if rewards:
                print('episode id: %d, mean reward: %.3f, steps/s: %.0f'
                      % (completed_episodes - 1, np.mean(rewards),
                         collector.steps / (time.perf_counter() - started)), flush=True)

            if rewards:
                print('%s success (last %d): %.1f%%' %
                      (task, min(100, len(SUCCESSES)), 100 * np.mean(SUCCESSES[-100:])), flush=True)

            if update_id % plot_every == 0:
                plot.update(REWARDS, reward_episode_start)
            else:
                plot.pump()  # keep the curve window responsive between refreshes

            if completed_episodes >= next_save:
                save_progress()
                next_save = (completed_episodes // save_every + 1) * save_every

            if render_every > 0 and completed_episodes >= next_render:
                plot.update(REWARDS, reward_episode_start)
                preview_reward = preview(net, task, max_steps, plot, **environment)
                print('preview reward: %.3f' % preview_reward, flush=True)
                next_render = (completed_episodes // render_every + 1) * render_every

    except KeyboardInterrupt:
        print('Stopped; saving model and reward curve.', flush=True)

    # Runs after normal completion or Ctrl+C, not after an unexpected training error.
    save_progress()
    if show_curve:
        plt.ioff()
        plt.show()  # leave the final curve available after training stops
