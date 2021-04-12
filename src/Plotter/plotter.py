import matplotlib.pyplot as plot
import numpy as np
from src.constants import *

def plot_average_reward(env, log_interval, running_reward):

    img_name = f"AverageReward_{env.algorithm_name}.png".replace(" ", "")

    plot.plot(np.arange(0, len(running_reward)*log_interval, log_interval), running_reward)
    plot.title(f'Average Reward with {env.algorithm_name}')
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    plot.savefig(img_name)
    plot.show()
    plot.close()

def plot_total_throughput(env):

    img_name = f"Throughpout_{env.algorithm_name}.png".replace(" ", "")

    plot.plot(env.operators[0].throughput, label = env.operators[0].name)
    plot.plot(env.operators[1].throughput, label = env.operators[1].name)
    plot.legend()
    plot.title(f'Total Throughput with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Throughput [Mb / s]')
    plot.savefig(img_name)
    plot.show()
    plot.close()

def plot_request_distribution(env):

    img_name = f"RequestDistribution_{env.algorithm_name}.png".replace(" ", "")

    plot.plot(env.operators[0].request_arr, label = env.operators[0].name)
    plot.plot(env.operators[1].request_arr, label = env.operators[1].name)
    plot.legend()
    plot.title(f'Request Distribution with {env.algorithm_name}')
    plot.xlabel('Timestep [t]')
    plot.ylabel('Request size [MHz * s]')
    plot.savefig(img_name)
    plot.show()
    plot.close()

def plot_packet_distribution(env):

    img_name = f"PacketDistribution_{env.algorithm_name}.png".replace(" ", "")

    plot.plot(env.operators[0].packet_distribution, label = env.operators[0].name)
    plot.plot(env.operators[1].packet_distribution, label = env.operators[1].name)
    plot.legend()
    plot.title('Packet Distribution')
    plot.xlabel('Timestep (t)')
    plot.ylabel('Number of packets')
    plot.savefig(img_name)
    plot.show()
    plot.close()