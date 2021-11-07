import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'

    max_m_episode = 800000
    max_steps = 800

    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            if episode_id % 100 == 1:
                env.render()

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



