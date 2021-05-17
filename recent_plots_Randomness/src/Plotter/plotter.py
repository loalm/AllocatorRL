import matplotlib.pyplot as plot
import numpy as np
from src.constants import *
import os
import torch
import torch.nn.functional as F

BASE_DIR = "recent_plots/"

if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

def plot_average_reward(env, log_interval, running_reward, show_plot=True):

    img_name = f"AverageReward_{env.algorithm_name}.png".replace(" ", "")

    plot.plot(np.arange(0, len(running_reward)*log_interval, log_interval), running_reward)
    plot.title(f'Average Reward with {env.algorithm_name}')
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()

def plot_quality_served_traffic(env, show_plot = True):

    img_name = f"QualityServedTraffic{env.algorithm_name}.png".replace(" ", "")

    plot.plot(env.operators[0].quality_served_traffic, label = env.operators[0].name)
    plot.plot(env.operators[1].quality_served_traffic, label = env.operators[1].name)
    plot.legend()
    plot.title(f'Quality Served Traffic with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Served Traffic [Mb / s]')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()

def plot_5th_percentile_throughput(env, show_plot = True):

    img_name = f"5thPercentileThroughput{env.algorithm_name}.png".replace(" ", "")

    plot.plot(env.operators[0].five_percentile_throughput, label = env.operators[0].name)
    plot.plot(env.operators[1].five_percentile_throughput, label = env.operators[1].name)
    plot.legend()
    plot.title(f'5% Worst Throughput with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Throughput [Mb / s]')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()

def plot_request_distribution(env, show_plot = True):

    img_name = f"RequestDistribution_{env.algorithm_name}.png".replace(" ", "")

    [o1, o2] = env.operators

    plot.plot(np.arange(TIMESTEPS)*T_SLOT, o1.request_arr, label = o1.name)
    plot.plot(np.arange(TIMESTEPS)*T_SLOT, o2.request_arr, label = o2.name)
    plot.legend()
    plot.title(f'Request Distribution with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Request size [MHz * s]')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()

    requests_softmaxed = [F.softmax(torch.tensor(requests).float(), dim=0) for requests in zip(o1.request_arr, o2.request_arr)]
    r1, r2 = zip(*requests_softmaxed)
    img_name = f"RequestDistribution_SOFTMAXED{env.algorithm_name}.png".replace(" ", "")
    plot.plot(r1, label = o1.name)
    plot.plot(r2, label = o2.name)
    plot.legend()
    plot.title(f'Softmaxed Request Distribution with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Request size [MHz * s]')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()



def plot_packet_distribution(env, show_plot = True):

    img_name = f"PacketDistribution_{env.algorithm_name}.png".replace(" ", "")
    
    for o in env.operators:
        packet_distribution = []
        for t in range(TIMESTEPS):
            packet_distribution.append(len(o.packets_at_timestep[t]))
            #
        plot.plot(np.arange(TIMESTEPS)*T_SLOT/60/60, packet_distribution, label = o.name)
        # print(f'Total packets: {sum(packet_distribution)}')
        # plot.plot(o.arrival_rates, label = o.name)
    #plot.plot(env.operators[1].packet_distribution, label = env.operators[1].name)
    plot.legend()
    plot.title('Packet Distribution')
    plot.xlabel('Hour of day')
    plot.ylabel('Number of packets')
    plot.savefig(BASE_DIR + img_name)
    if show_plot:
        plot.show()
    plot.close()

def plot_reward_timestep(env, reward_timestep, show_plot = True):
    plot.plot(reward_timestep)
    # plot.legend()
    plot.title(f'Reward at timestep t with {env.algorithm_name}')
    plot.xlabel('Timestep t')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig(BASE_DIR + f'Reward_Timestep{env.algorithm_name}_BW{env.bandwidth}_TS{TIMESTEPS}.png')
    if show_plot:
        plot.show()
    plot.close()

def plot_action_timestep(env, action_timestep, show_plot = True):
    plot.plot(action_timestep)
    # plot.legend()
    plot.title(f'Spectrum Allocated to Operator 1 at timestep t with {env.algorithm_name}')
    plot.xlabel('Timestep t')
    plot.ylabel(r'$a_1$')
    plot.savefig(BASE_DIR + f'Action_Timestep{env.algorithm_name}_BW{env.bandwidth}_TS{TIMESTEPS}.png')
    if show_plot:
        plot.show()
    plot.close()