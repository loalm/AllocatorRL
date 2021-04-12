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

parse = argparse.ArgumentParser(description='PyTorch baseline')
parse.add_argument('--split', type=float, default=0.5, metavar='N')
parse.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parse.parse_args()

env = Environment()

def baseline_softmax(episodes=100, bandwidth=None):
    NUM_EPISODES = episodes
    running_reward = []
    avg_reward = 0.0
    env.algorithm_name = f"Baseline Softmax Algorithm"
    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            state = torch.tensor(state, dtype=float)
            action = F.softmax(state)
            state, reward = env.step(action, t)
            ep_reward += reward

        avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(ep_reward)

    avg_reward /= NUM_EPISODES

    plot_average_reward(env, args.log_interval, running_reward)
    plot_packet_distribution(env)
    plot_request_distribution(env)
    plot_total_throughput(env)

    return avg_reward

def baseline(episodes=500, bandwidth=None):
    NUM_EPISODES = episodes
    running_reward = []
    avg_reward = 0.0
    env.algorithm_name = f"Baseline {args.split} Algorithm"
    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            action = [args.split, 1-args.split] # Split evenly
            state, reward = env.step(action, t)
            ep_reward += reward

        avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            #print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(ep_reward)

    avg_reward /= NUM_EPISODES

    plot_average_reward(env, args.log_interval, running_reward)
    plot_packet_distribution(env)
    plot_request_distribution(env)
    plot_total_throughput(env)

    return avg_reward

def spectrum_baseline():

    spectrum = np.arange(0, 100, 10)
    rewards = []

    for s in spectrum:
        print(f"Spectrum s: {s}")
        rewards.append(baseline(episodes = 50, bandwidth = s))

    print(rewards)

    plot.plot(spectrum, rewards)
    plot.title(f'Average Reward with Baseline {args.split} / {1-args.split} Algorithm and Spectrum Size s')
    plot.xlabel('Bandwidth [MHz]')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig('baseline_avg_reward_spectrum.png')
    plot.show()
    plot.close()


if __name__ == '__main__':
    baseline_softmax()
    #baseline()
    #spectrum_baseline()