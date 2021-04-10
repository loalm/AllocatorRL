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

parse = argparse.ArgumentParser(description='PyTorch baseline')
parse.add_argument('--split', type=float, default=0.5, metavar='N')
parse.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
arg = parse.parse_args()

env = Environment()

def baseline_softmax(episodes=500, bandwidth=None):
    NUM_EPISODES = episodes
    running_reward = []
    avg_reward = 0.0
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
        if i_episode % arg.log_interval == 0:
            print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(ep_reward)

    avg_reward /= NUM_EPISODES

    plot.plot(np.arange(0, NUM_EPISODES, arg.log_interval), running_reward)
    plot.title(f'Average Reward with Baseline SOFTMAX Algorithm')
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig('baseline_softmax.png')
    plot.show()
    plot.close()

    # plot.plot(env.operators[0].packet_distribution, label = env.operators[0].name)
    # plot.plot(env.operators[1].packet_distribution, label = env.operators[1].name)
    # plot.legend()
    # plot.title('Packet Distribution')
    # plot.xlabel('Timestep (t)')
    # plot.ylabel('Number of packets')
    # plot.savefig('packet_distribution.png')
    # plot.show()
    # plot.close()

    plot.plot(env.operators[0].request_arr, label = env.operators[0].name)
    plot.plot(env.operators[1].request_arr, label = env.operators[1].name)
    plot.legend()
    plot.title('Request Distribution')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Request size [MHz * s]')
    plot.savefig('baseline_softmax_request_distribution.png')
    plot.show()
    plot.close()

    return avg_reward

def baseline(episodes=500, bandwidth=None):
    NUM_EPISODES = episodes
    running_reward = []
    avg_reward = 0.0
    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            action = [arg.split, 1-arg.split] # Split evenly
            state, reward = env.step(action, t)
            ep_reward += reward

        avg_reward += ep_reward
        if i_episode % arg.log_interval == 0:
            #print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(ep_reward)

    avg_reward /= NUM_EPISODES

    plot.plot(np.arange(0, NUM_EPISODES, arg.log_interval), running_reward)
    plot.title(f'Average Reward with Baseline {arg.split} / {1-arg.split} Algorithm')
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig('baseline1.png')
    plot.show()
    plot.close()

    # plot.plot(env.operators[0].packet_distribution, label = env.operators[0].name)
    # plot.plot(env.operators[1].packet_distribution, label = env.operators[1].name)
    # plot.legend()
    # plot.title('Packet Distribution')
    # plot.xlabel('Timestep (t)')
    # plot.ylabel('Number of packets')
    # plot.savefig('packet_distribution.png')
    # plot.show()
    # plot.close()

    plot.plot(env.operators[0].request_arr, label = env.operators[0].name)
    plot.plot(env.operators[1].request_arr, label = env.operators[1].name)
    plot.legend()
    plot.title('Request Distribution')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Request size [MHz * s]')
    plot.savefig('baseline_request_distribution.png')
    plot.show()
    plot.close()

    return avg_reward

def spectrum_baseline():

    spectrum = np.arange(0, 100, 10)
    rewards = []

    for s in spectrum:
        print(f"Spectrum s: {s}")
        rewards.append(baseline(episodes = 50, bandwidth = s))

    print(rewards)

    plot.plot(spectrum, rewards)
    plot.title(f'Average Reward with Baseline {arg.split} / {1-arg.split} Algorithm and Spectrum Size s')
    plot.xlabel('Bandwidth [MHz]')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig('baseline_avg_reward_spectrum.png')
    plot.show()
    plot.close()


if __name__ == '__main__':
    #baseline_softmax()
    baseline()
    #spectrum_baseline()