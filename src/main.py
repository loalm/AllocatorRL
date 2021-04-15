import argparse
from src.Reinforce import reinforce
from src.Test.baseline import baseline, baseline_softmax
import matplotlib.pyplot as plot
import numpy as np



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


def main():
    reinforce_reward, env1 = reinforce.main(args)
    baseline_reward, env2 = baseline(args)
    baseline_softmax_reward, env3 = baseline_softmax(args)
    baseline_skewed_reward, env4 = baseline(args, split=0.1)
    env4.algorithm_name = "Baseline skewed 0.1/0.9"

    episode_intervals = np.arange(0, args.episodes, args.log_interval)
    plot.plot(episode_intervals, reinforce_reward, label = env1.algorithm_name)
    plot.plot(episode_intervals, baseline_reward, label = env2.algorithm_name)
    plot.plot(episode_intervals, baseline_softmax_reward, label = env3.algorithm_name)
    plot.plot(episode_intervals, baseline_skewed_reward, label = env4.algorithm_name)
    plot.legend()
    plot.title(f'Average Reward (Quality Traffic Served)')
    plot.xlabel('Episode')
    plot.ylabel('Reward [Mb / s]')
    img_name = f"AverageRewardQTS_all.png".replace(" ", "")
    plot.savefig(img_name)
    plot.show()
    plot.close()



if __name__ == '__main__':
    main()