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
import math

parser = argparse.ArgumentParser(description='PyTorch baseline')
parser.add_argument('--split', type=float, default=0.5, metavar='N')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--episodes', type=int, default=300, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--show-plot', action='store_true', help='Show the plots')
args = parser.parse_args()


def baseline_weighted(args, bandwidth=None):
    episodes = 50#args.episodes
    running_reward = []
    env = Environment()
    env.algorithm_name = f"Baseline WeightedRequest"
    print(f"Running {env.algorithm_name}")

    action_timestep = np.array([0]*TIMESTEPS, dtype='float')


    interval_avg_reward = 0
    for i_episode in range(episodes):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            # [r1, r2] = state
            [o1, o2] = env.operators
            # [r1, r2] = [o1.request, o2.request]
            # print(f"arr1: {state}")
            # print(f"arr2: {[o1.request, o2.request]}")
            requests = []
            n_datapoints = 5
            if t > n_datapoints:
                x = [*range(t-n_datapoints,t)]
                y = [np.array(o.request_arr)[x] for o in env.operators]
                request_polyfits = [np.polyfit(x, y[i], deg=2) for i, _ in enumerate(env.operators)]
                request_polyvals = [np.polyval(request_polyfits[i], t) for i, _ in enumerate(env.operators)]
                requests = request_polyvals
            else:
                requests = [o.request for o in env.operators]   

            if all([val == 0 for val in requests]):
                action = [.5]*len(env.operators)
            else:
                summ = sum(requests)
                action = [requests[i] / summ for i, _ in enumerate(env.operators)]

            # state = torch.tensor(state, dtype=float)
            # action = F.softmax(state)
            state, reward = env.step(action, t)

            reward = math.exp(reward)


            ep_reward += reward
            if i_episode == episodes - 1:
                print(f"t: {t} Action: {action}")
                action_timestep[t] = action[0]

        interval_avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval
            print(f'Episode {i_episode} \t Last reward: {interval_avg_reward} \t Average reward: {running_reward}')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0


    plot_action_timestep(env, action_timestep, show_plot=args.show_plot)
    plot_5th_percentile_throughput(env, show_plot=args.show_plot)
    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)

    return running_reward, env

def baseline_tpmax(args, bandwidth=None):
    episodes = args.episodes
    running_reward = []
    env = Environment()
    env.algorithm_name = f"Baseline TPMAX"
    print(f"Running {env.algorithm_name}")

    interval_avg_reward = 0
    for i_episode in range(episodes):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            # [r1, r2] = state
            [o1, o2] = env.operators
            # [r1, r2] = [o1.request, o2.request]
            # print(f"arr1: {state}")
            # print(f"arr2: {[o1.request, o2.request]}")
            if o1.five_percentile_throughput[t-1] == o2.five_percentile_throughput[t-1]:
                action = [.5,.5]
            elif o1.five_percentile_throughput[t-1] > o2.five_percentile_throughput[t-1]:
                action = [0.45,0.55]
            else:
                action = [0.55,0.45]
            # state = torch.tensor(state, dtype=float)
            # action = F.softmax(state)
            state, reward = env.step(action, t)
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
    episodes = 50#args.episodes
    running_reward = []
    env = Environment()
    env.algorithm_name = f"Baseline {split}"
    interval_avg_reward = 0

    print(f"Running {env.algorithm_name}")

    util_sum = np.zeros((2, episodes))

    for i_episode in range(episodes):
        state = env.reset_state()
        if bandwidth is not None:
            env.bandwidth = bandwidth
        ep_reward = 0
        for t in range(TIMESTEPS):
            action = [split, 1-split] # Split evenly
            state, reward = env.step(action, t)
            ####
            reward = math.exp(reward)
            ####
            ep_reward += reward
            
        interval_avg_reward += ep_reward
        util_sum[0][i_episode] = sum(env.operators[0].utilisation)
        util_sum[1][i_episode] = sum(env.operators[1].utilisation)

        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval
            print(f'Episode {i_episode} \t Last reward: {interval_avg_reward}')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0


    util_sum[0] = (util_sum[0] / TIMESTEPS) * 100
    util_sum[1] = (util_sum[1] / TIMESTEPS) * 100

    plot.plot(util_sum[0], label = env.operators[0].name)
    plot.plot(util_sum[1], label = env.operators[1].name)
    plot.legend()
    plot.title(f'Time Utilization per episode with {env.algorithm_name} algorithm')
    plot.xlabel('Episode (i_episode)')
    plot.ylabel('Utilization [%]')
    plot.savefig(f'Time_Utilization_{env.algorithm_name}_BW{env.bandwidth}_TS{TIMESTEPS}.png')
    #plot.show()
    plot.close()

    plot_5th_percentile_throughput(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)
    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)

    return running_reward, env

def spectrum_baseline(args):

    spectrum = np.arange(10, 150, 10)
    rewards = []

    for s in spectrum:
        print(f"Spectrum s: {s}")
        reward, env = baseline(args, bandwidth = s)
        rewards.append(reward)
        print(f"Reward: {reward}")

    print(rewards)

    plot.plot(spectrum, rewards)
    plot.title(f'Average Reward with Baseline {args.split} / {1-args.split} Algorithm and Bandwidth b')
    plot.xlabel('Bandwidth [MHz]')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig(f'{env.algorithm_name}_avg_reward_spectrum.png')
    plot.show()
    plot.close()


if __name__ == '__main__':
    #baseline_softmax(args)
    #baseline(args)
    spectrum_baseline(args)