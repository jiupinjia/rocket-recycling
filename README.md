# Rocket-recycling with Reinforcement Learning

Developed by: [Zhengxia Zou, Ph.D.](https://zhengxiazou.github.io/)




## One-min demo video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/gsIiniJMr3E/0.jpg)](https://www.youtube.com/watch?v=gsIiniJMr3E)



## About this project

As a big fan of SpaceX, I always dreamed of having my own rockets. Recently, I worked on an interesting question that whether we can "build" a virtual rocket and address a challenging problem - rocket recycling, with simple reinforcement learning. 

I tried on two tasks: **hovering** and **landing**. The rocket is simplified into a rigid body on a 2D plane. I considered the basic cylinder dynamics model and assumed the air resistance is proportional to the velocity. A thrust-vectoring engine is installed at the bottom of the rocket. This engine provides adjustable thrust values (0.2g, 1.0g, and 2.0g) with different directions. An angular velocity constraint is added to the nozzle with a max-rotating speed of 30 degrees/second.

With the above basic settings, the action space is defined as a collection of the discrete control signals of the engine, including the thrust acceleration and the angular velocity of the nozzle. The state-space consists of the rocket position, speed, angle, angle velocity, nozzle angle, and the simulation time.

![](./gallery/config.jpg)



For the landing task, I followed the basic parameters of the Starship SN10 belly flop maneuver. The initial speed is set to -50m/s. The rocket orientation is set to 90 degrees (horizontally). The landing burn height is set to 500 meters above the ground. 

![](./gallery/timelapse.jpg)

Image credit https://twitter.com/thejackbeyer/status/1367364251233497095



The reward functions are quite straightforward.

For the hovering tasks: the step-reward is given based on two rules: 1) The distance between the rocket and the predefined target point - the closer they are, the larger reward will be assigned. 2) The angle of the rocket body (the rocket should stay as upright as possible)

For the landing task: we look at the Speed and angle at the moment of contact with the ground - when the touching-speed are smaller than a safe threshold and the angle is close to 0 degrees (upright), we see it as a successful landing and a big reward will be assigned. The rest of the rules are the same as the hovering task.


I implement the above environment and train a policy-based agent (actor-critic) to solve this problem. The episode reward finally converges very well after over 20000 training episodes.

| Fully trained agent (task: hovering) |        Reward over number of episodes        |
| :----------------------------------: | :------------------------------------------: |
|       ![](./gallery/h_20k.gif)       | ![](./gallery/hovering_rewards_00022301.jpg) |


| Fully trained agent (task: landing) |        Reward over number of episodes        |
| :----------------------------------: | :------------------------------------------: |
|       ![](./gallery/l_11k.gif)       | ![](./gallery/landing_rewards_00011201.jpg) |


Despite the simple setting of the environment and the reward, the agent has learned the belly flop maneuver nicely. The following animation shows a comparison between the real SN10 and a fake one learned from reinforcement learning.

![](./gallery/belly_flop.gif)




## Requirements

See [Requirements.txt](Requirements.txt).



## Usage

To train an agent, see `./example_train.py`

To test an agent:

```python
import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break
```



## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">  Rocket-recycling</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://www-personal.umich.edu/~zzhengxi/">Zhengxia Zou</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.



## Citation

``````
@misc{zou2021rocket,
  author = {Zhengxia Zou},
  title = {Rocket-recycling with Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jiupinjia/rocket-recycling}}
}
``````