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


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.90, metavar='G',
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
            nn.Softmax(dim = 0),
            nn.Linear(2, 32),
            #nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim = 0)
        )

    def forward(self, x):
        # x = self.affine1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # action_scores = self.affine2(x)
        # return F.softmax(action_scores.clone(), dim=1)
        x = self.net(x)
        #x = torch.exp(x)
        #x = F.log_softmax(x)
        return x
        #return self.net(x)
        #return Variable(self.net(x),  requires_grad=True)

policy = Policy()
# Best lr : 1e-2 = 0.01 ... 0.05
optimizer = optim.Adam(policy.parameters(), lr=0.01) # TODO: Adjust learning rate
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    #print(f"RETURNS state: {state}")
    #out = policy(Variable(state))
    out = policy(state)
    #print(f"Out: {out}")
    d = Dirichlet(out+eps) # Add epsilon to make sure no out value is equal to Dirichlet lowerbound 0.
    action = d.sample()
    #print(f"RETURNS log prob: {d.log_prob(action)}")

    policy.saved_log_probs.append(d.log_prob(action))
    return action.squeeze(0).numpy()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    #print(f'Saved_log_probs {policy.saved_log_probs}')
    for log_prob, R in zip(policy.saved_log_probs, returns):
       policy_loss.append(-log_prob * R)

    policy_loss = torch.stack(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    #policy.saved_log_probs = Variable(torch.Tensor())

def main(args):
    NUM_EPISODES = args.episodes
    running_reward = []
    interval_avg_reward = 0

    action_mem = np.array([0]*NUM_EPISODES, dtype='float')
    
    env.algorithm_name = "REINFORCE Algorithm"

    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        ep_reward = 0
        #print("\n\n")
        for t in range(TIMESTEPS):
            #print(f"State: {state}")
            action = select_action(state)
            
            #print(f"t: {t} Action: {action}")
            state, reward = env.step(action, t)
            policy.rewards.append(reward)
            ep_reward += reward
            if i_episode == NUM_EPISODES - 1:
                print(f"t: {t} Action: {action}")
   
            action_mem[i_episode] = action[0]

        interval_avg_reward += ep_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            if i_episode != 0:
                interval_avg_reward /= args.log_interval

            print(f'Episode {i_episode} \t Last reward: {interval_avg_reward} ')
            running_reward.append(interval_avg_reward)
            interval_avg_reward = 0

    #action_mem /= TIMESTEPS

    # plot.plot(np.arange(0, NUM_EPISODES, args.log_interval), running_reward)
    # plot.title('Average Reward per Training Episode')
    # plot.xlabel('Episode')
    # plot.ylabel('Running Reward [Mb / s]')
    # plot.savefig('avg_reward.png')
    # plot.show()
    # plot.close()

    plot.plot(action_mem)
    plot.title('Average Spectrum % Allocated to Operator 1 per training Episode')
    plot.xlabel('Episode (i_episode)')
    plot.ylabel('Spectrum %')
    plot.savefig('avg_allocation.png')
    #plot.show()
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

    plot_5th_percentile_throughput(env, show_plot=args.show_plot)

    plot_average_reward(env, args.log_interval, running_reward, show_plot=args.show_plot)
    plot_packet_distribution(env, show_plot=args.show_plot)
    plot_request_distribution(env, show_plot=args.show_plot)
    plot_quality_served_traffic(env, show_plot=args.show_plot)

    return running_reward, env



if __name__ == '__main__':
    main(args)