import argparse
from src.Reinforce import reinforce, actor_critic
from src.Plotter.plotter import BASE_DIR
from src.Test.baseline import baseline, baseline_tpmax, baseline_weighted
import matplotlib.pyplot as plot
import numpy as np
from src.constants import *



parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--show-plot', action='store_true',
                    help='Show the plots')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--split', type=float, default=0.5, metavar='N')

parser.add_argument('--episodes', type=int, default=500, metavar='N',
help='Number of training episodes (default: 500)')
args = parser.parse_args()

def main_spectrumtest():
    
    num_cells = 5
    spectrum = SPECTRUM
    rewards = {}

    arrival_rates1 = (np.sin(x)*AMPLITUDES[0]).astype(int) + PACKETS_PER_OPERATOR_PER_SECOND#//60
    arrival_rates2 = (np.sin(x)*AMPLITUDES[1]).astype(int) + PACKETS_PER_OPERATOR_PER_SECOND#//60


    for s in spectrum:
        print(f'Spectrum : {s}')
        reward_env = [
                    reinforce.main(args, bandwidth=s*num_cells),
                    baseline(args, bandwidth=s*num_cells),
                    baseline_weighted(args, bandwidth=s*num_cells),
                    #   actor_critic.main(args, bandwidth=s), 
                    #   baseline_tpmax(args, bandwidth=s),
                    #   baseline(args, split=0.1, bandwidth=s),
                    #   baseline(args, split=0.9, bandwidth=s)
                    ]

        for (reward, env) in reward_env:
            if env.algorithm_name == 'Baseline 0.5':
                reward = sum(reward)/len(reward)
            else:
                reward = max(reward)
            reward /= num_cells
            if env.algorithm_name not in rewards:
                rewards[env.algorithm_name] = [reward]
            else:
                rewards[env.algorithm_name].append(reward)
    
        print(rewards)
    for algo, reward in rewards.items():
        plot.plot(spectrum, reward, label = algo)

    plot.legend()
    plot.suptitle("Reward (Quality Served Traffic)")
    #plot.title(f"Allocator Bandwidth: {env1.bandwidth} MHz Timesteps: {TIMESTEPS}", fontsize=10)
    plot.xlabel('Bandwidth [MHz]')
    plot.ylabel('Reward [Mb / s]')
    img_name = BASE_DIR+f"RewardAtBandWidth_all.png".replace(" ", "")
    plot.savefig(img_name)
    plot.show()
    plot.close()


def main():
    reinforce_reward, env1 = reinforce.main(args)
    baseline_reward, env2 = baseline(args)
    baseline_softmax_reward, env3 = baseline_tpmax(args)
    baseline_skewed_reward, env4 = baseline(args, split=0.1)
    baseline_skewed_reward2, env5 = baseline(args, split=0.9)
    # ac_reward, env6 = actor_critic.main(args)

    env4.algorithm_name = "Baseline skewed 0.1/0.9"
    env5.algorithm_name = "Baseline skewed 0.9/0.1"

    episode_intervals = np.arange(0, args.episodes, args.log_interval)
    plot.plot(episode_intervals, reinforce_reward, label = env1.algorithm_name)
    plot.plot(episode_intervals, baseline_reward, label = env2.algorithm_name)
    plot.plot(episode_intervals, baseline_softmax_reward, label = env3.algorithm_name)
    plot.plot(episode_intervals, baseline_skewed_reward, label = env4.algorithm_name)
    plot.plot(episode_intervals, baseline_skewed_reward2, label = env5.algorithm_name)
    # plot.plot(episode_intervals, ac_reward, label = env6.algorithm_name)

    plot.legend()
    plot.suptitle("Reward (Quality Traffic Served)")
    plot.title(f"Allocator Bandwidth: {env1.bandwidth} MHz Timesteps: {TIMESTEPS}", fontsize=10)
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    img_name = BASE_DIR+f"AverageRewardQTS_all_BW{env1.bandwidth}_TS{TIMESTEPS}.png".replace(" ", "")
    plot.savefig(img_name)
    plot.show()
    plot.close()



if __name__ == '__main__':
    # main()
    main_spectrumtest()