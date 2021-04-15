import argparse
from src.Reinforce.environment import Environment
from src.constants import TIMESTEPS
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from itertools import count
import matplotlib.pyplot as plot
from src.Plotter.plotter import *

parser = argparse.ArgumentParser(description='PyTorch baseline')
parser.add_argument('--split', type=float, default=0.5, metavar='N')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--episodes', type=int, default=300, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--show-plot', action='store_true', help='Show the plots')
args = parser.parse_args()


def baseline_softmax(args, bandwidth=None):
    episodes = args.episodes
    running_reward = []
    env = Environment()
    env.algorithm_name = f"Baseline Softmax Algorithm"
    interval_avg_reward = 0
    for i_episode in range(episodes):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            state = torch.tensor(state, dtype=float)
            action = F.softmax(state)
            _, reward = env.step(action, t)
            ep_reward += reward

        interval_avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval
            print(f'Episode {i_episode} \t Last reward: {interval_avg_reward} \t Average reward: {running_reward}')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0

    plot_5th_percentile_throughput(env, show_plot=args.show_plot)
    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)

    return running_reward, env


def baseline(args, bandwidth=None, split=args.split):
    episodes = args.episodes
    running_reward = []
    env = Environment()
    env.algorithm_name = f"Baseline {split} Algorithm"
    interval_avg_reward = 0
    for i_episode in range(episodes):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            action = [split, 1-split] # Split evenly
            state, reward = env.step(action, t)
            ep_reward += reward

        interval_avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval
            #print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0

    plot_5th_percentile_throughput(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)
    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)

    return running_reward, env

def spectrum_baseline(args):

    spectrum = np.arange(10, 100, 10)
    rewards = []

    for s in spectrum:
        print(f"Spectrum s: {s}")
        reward, _ = baseline(args, bandwidth = s)
        rewards.append(reward)

    print(rewards)

    plot.plot(spectrum, rewards)
    plot.title(f'Average Reward with Baseline {args.split} / {1-args.split} Algorithm and Spectrum Size s')
    plot.xlabel('Bandwidth [MHz]')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig('baseline_avg_reward_spectrum.png')
    plot.show()
    plot.close()


if __name__ == '__main__':
    #baseline_softmax(args)
    #baseline(args)
    spectrum_baseline(args)