from pathlib import Path
import os

import cv2
import torch

from ppo import PPOActorCritic
from rocket import Rocket


if __name__ == '__main__':

    task = 'landing'  # 'hover' or 'landing'; match the training task
    max_steps = 800
    wait_time = 25  # two rendered frames per step: approximately real time
    ckpt_path = Path(task + '_ppo_quadratic_v1_ckpt') / 'ppo_latest.pt'

    os.chdir(Path(__file__).resolve().parent)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if checkpoint.get('algorithm') != 'ppo' or checkpoint.get('task') != task:
        raise ValueError('Checkpoint algorithm/task does not match this inference configuration')

    env = Rocket(task=task, max_steps=max_steps, **checkpoint['environment'])
    net = PPOActorCritic(task).to(device)
    net.load_state_dict(checkpoint['model_G_state_dict'])
    net.eval()

    try:
        state = env.reset()
        with torch.no_grad():
            for step_id in range(max_steps):
                action, _, _ = net.get_action(state)
                state, reward, done, _ = env.step(action)
                env.render(window_name='Rocket recycling - inference', wait_time=wait_time)
                if done:
                    break
        cv2.waitKey(1000)
    finally:
        cv2.destroyAllWindows()
