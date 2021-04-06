import argparse
from src.Reinforce.environment import Environment
from src.constants import TIMESTEPS
import src.constants as constants
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from itertools import count
from torch.autograd import Variable
import matplotlib.pyplot as plot


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
#parser.add_argument('--render', action='store_true',
#                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

np.random.seed(args.seed)
env = Environment()
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.saved_log_probs = []
        self.rewards = []

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        # x = self.affine1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # action_scores = self.affine2(x)
        # #action_scores = torch.exp(action_scores)
        # return F.softmax(action_scores.clone(), dim=1)

        return Variable(self.net(x),  requires_grad=True)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4) # TODO: Adjust learning rate
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    #print(f"RETURNS state: {state}")
    out = policy(Variable(state))
    #print(f"RETURNS out: {out}")
    #out += eps # To make sure no out value is equal to Dirichlet lowerbound 0.
    d = Dirichlet(out+eps)
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

def plot_xy(x, y):
    print("Plotting...")
    # Get x values of the sine wave
    #time = np.arange(0, 10, 0.1);
    # Amplitude of the sine wave is sine of a variable like time
    #amplitude = np.sin(time)
    # Plot a sine wave using time and amplitude obtained for the sine wave
    plot.plot(x, y)
    plot.show()

def main():
    NUM_EPISODES = 100
    running_reward = [10]
    for i_episode in range(NUM_EPISODES):
        state = env.reset_state()
        ep_reward = 0
        for t in range(TIMESTEPS):
            action = select_action(state)
            state, reward = env.step(action, t)
            policy.rewards.append(reward)
            ep_reward += reward

        #running_reward[-1] = 0.05 * ep_reward + (1 - 0.05) * running_reward[-1]
        running_reward[-1] = ep_reward/TIMESTEPS
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode} \t Last reward: {ep_reward} \t Average reward: {running_reward}')
            running_reward.append(running_reward[-1])

    plot.plot(np.arange(0, NUM_EPISODES+args.log_interval, args.log_interval), running_reward)
    plot.title('Average Reward per Training Episode')
    plot.xlabel('Episode')
    plot.ylabel('Running Reward')
    plot.savefig('avg_reward.png')
    plot.show()
    plot.close()

    # plot.plot(env.operators[0].packet_distribution)
    # plot.plot(env.operators[1].packet_distribution)
    # plot.title('Packet distribution')
    # plot.xlabel('Timestep (t)')
    # plot.ylabel('Number of packets')
    # plot.savefig('packet_distribution.png')
    # plot.show()
    # plot.close()


if __name__ == '__main__':
    main()



