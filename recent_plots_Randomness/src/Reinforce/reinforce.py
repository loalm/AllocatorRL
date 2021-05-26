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
import time


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.5, metavar='G', 
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
#parser.add_argument('--render', action='store_true',
#                    help='render the environment')
parser.add_argument('--show-plot', action='store_true',
                    help='Show the plots')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--episodes', type=int, default=500, metavar='N',
help='Number of training episodes (default: 500)')
args = parser.parse_args()

np.random.seed(args.seed)
env = Environment()
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.saved_log_probs = []
        self.rewards = []

        self.net = nn.Sequential(
            # nn.Softmax(dim = 0),
            nn.Linear(4, 2),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


policy = Policy()
# Best lr : 1e-2 = 0.01 ... 0.05
#0.0005
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE) # TODO: Adjust learning rate
# optimizer = optim.SGD(policy.parameters(), lr=0.0001, momentum=0.99)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    #print(f"RETURNS state: {state}")
    #out = policy(Variable(state))
    out = policy(state)
    #print(f"Out: {out+eps}")
    d = Dirichlet(out+eps) # Add epsilon to make sure no out value is equal to Dirichlet lowerbound 0.
    action = d.sample()
    # print(f"action: {action}")
    #print(f"RETURNS log prob: {d.log_prob(action)}")
    policy.saved_log_probs.append(d.log_prob(action))
    return action.squeeze(0).numpy()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        # R = -R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # print(returns)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        # print(R)
        policy_loss.append(-log_prob * R)
    
   # print(f'policy_loss {policy_loss} \n')
    policy_loss = torch.stack(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main(args, bandwidth=None):
    if bandwidth:
        env.bandwidth = bandwidth
    NUM_EPISODES = args.episodes
    running_reward = []
    interval_avg_reward = 0

    action_timestep = np.array([0]*TIMESTEPS, dtype='float')
    policy.load_state_dict(torch.load('policy_weights.pth'))
    env.algorithm_name = "REINFORCE"
    print(f"Running {env.algorithm_name}")

    util_sum = np.zeros((2, NUM_EPISODES))
    reward_timestep = np.zeros(TIMESTEPS)
    eta = 1.1#.5#.7
    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        ep_reward = 0
        #print("\n\n")
        # start = time.time()
        for t in range(TIMESTEPS):
            #print(f"State: {state}")
            if np.random.rand() < eta:
                action = select_action(state)
            else:
                x = np.random.rand()
                action = [x, 1-x]
            #print(f"t: {t} Action: {action}")
            action = action[:2] # Test
            state, reward = env.step(action, t)
            policy.rewards.append(reward)
            # reward = math.exp(reward)
            ep_reward += reward
            if i_episode == NUM_EPISODES - 1:
                # print(f"t: {t} Action: {action}")
                reward_timestep[t] = reward
                action_timestep[t] = action[0]
        # end = time.time()
        # print("Scheduler Time elapsed: ", end-start)
        finish_episode()
        eta /= 0.95
        #print(eta)

        util_sum[0][i_episode] = sum(env.operators[0].utilisation)
        util_sum[1][i_episode] = sum(env.operators[1].utilisation)
        interval_avg_reward += ep_reward
        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval    


            print(f'Episode {i_episode} \t Last reward: {interval_avg_reward} ')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0

    util_sum[0] = (util_sum[0] / TIMESTEPS) * 100
    util_sum[1] = (util_sum[1] / TIMESTEPS) * 100

    # plot.plot(reward_timestep)
    # # plot.legend()
    # plot.title(f'Reward at timestep t with {env.algorithm_name}')
    # plot.xlabel('Timestep t')
    # plot.ylabel('Reward [Mb / s]')
    # plot.savefig(f'Reward_Timestep{env.algorithm_name}_BW{env.bandwidth}_TS{TIMESTEPS}.png')
    # plot.show()
    # plot.close()


    plot.plot(util_sum[0], label = env.operators[0].name)
    plot.plot(util_sum[1], label = env.operators[1].name)
    plot.legend()
    plot.title(f'Time Utilization per episode with {env.algorithm_name}')
    plot.xlabel('Episode (i_episode)')
    plot.ylabel('Utilization [%]')
    plot.savefig(f'Time_Utilization_{env.algorithm_name}_BW{env.bandwidth}_TS{TIMESTEPS}.png')
    #plot.show()
    plot.close()

    plot.plot(action_timestep)
    plot.title('Average Spectrum % Allocated to Operator 1 during final episode')
    plot.xlabel('Timestep')
    plot.ylabel(r'$a_1$')
    plot.savefig('avg_allocation.png')
    #plot.show()
    plot.close()

    # plot_action_timestep(env, action_timestep, show_plot=args.show_plot)
    plot_reward_timestep(env, reward_timestep, show_plot=args.show_plot)
    plot_5th_percentile_throughput(env, show_plot=args.show_plot)
    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)

    torch.save(policy.state_dict(), "policy_weights.pth")

    return running_reward, env



if __name__ == '__main__':
    main(args)