"""Live reward curve and periodic evaluation preview for example_train.py."""
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from rocket import Rocket


class RewardPlot:
    def __init__(self, task, show=True):
        self.show = show
        if show:
            plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(9, 5))
        self.raw, = self.axes.plot([], [], alpha=.35, linewidth=.7, label='episode reward')
        self.average, = self.axes.plot([], [], linewidth=1.8, label='moving avg (50)')
        self.axes.set(xlabel='completed episode', ylabel='reward', title=task + ' training')
        self.axes.legend(loc=2)
        self.axes.grid(alpha=.2)
        self.figure.tight_layout()
        if show:
            plt.show(block=False)

    def update(self, rewards, first_episode=0):
        values = np.asarray(rewards, dtype=float)
        episodes = np.arange(first_episode + 1, first_episode + len(values) + 1)
        stride = max(1, len(values) // 5000)
        self.raw.set_data(episodes[::stride], values[::stride])
        if len(values):
            cumulative = np.r_[0., np.cumsum(values)]
            ends = np.arange(1, len(values) + 1)
            starts = np.maximum(0, ends - 50)
            average = (cumulative[ends] - cumulative[starts]) / (ends - starts)
            self.average.set_data(episodes[::stride], average[::stride])
        self.axes.relim()
        self.axes.autoscale_view()
        self.figure.canvas.draw_idle()
        self.pump()

    def pump(self):
        if self.show and plt.fignum_exists(self.figure.number):
            self.figure.canvas.flush_events()
            plt.pause(.001)

    def save(self, path):
        self.figure.savefig(path, dpi=140)


@torch.no_grad()
def preview(net, task, max_steps, plot=None, wait_time=10, landing_reward="legacy",
            drag_model="quadratic", drag_coefficient=.004, hover_reward="legacy"):
    """One independent episode; preserve training RNG streams and model mode."""
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    was_training = net.training
    window_name = 'Rocket recycling - policy preview'
    env = None
    try:
        net.eval()
        env = Rocket(max_steps=max_steps, task=task, landing_reward=landing_reward,
                     drag_model=drag_model, drag_coefficient=drag_coefficient, hover_reward=hover_reward)
        state = env.reset()
        total_reward = 0.
        for step in range(max_steps):
            # Sample the learned policy without the old extra random exploration.
            action, _, _ = net.get_action(state, exploration=0.)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render(window_name=window_name, wait_time=wait_time)
            if plot is not None and step % 10 == 0:
                plot.pump()
            if done:
                break
        return total_reward
    finally:
        if env is not None:
            cv2.destroyAllWindows()
        net.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
