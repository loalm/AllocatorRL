import argparse
import numpy as np
from itertools import count
from collections import namedtuple
from src.Reinforce.environment import Environment
from src.constants import TIMESTEPS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1.01, metavar='G', 
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


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        # self.action_head = nn.Linear(128, 2)

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softplus()
        )

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        action_prob = self.net(x)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(),lr=0.01)
eps = np.finfo(np.float32).eps.item()


# def select_action(state):
#     state = torch.from_numpy(state).float()
#     probs, state_value = model(state)

#     # create a categorical distribution over the list of probabilities of actions
#     m = Categorical(probs)

#     # and sample an action using the distribution
#     action = m.sample()

#     # save to action buffer
#     model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

#     # the action to take (left or right)
#     return action.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    #print(f"RETURNS state: {state}")
    #out = policy(Variable(state))
    # out = policy(state)
    #print(f"Out: {out+eps}")
    d = Dirichlet(probs+eps) # Add epsilon to make sure no out value is equal to Dirichlet lowerbound 0.
    action = d.sample().to(torch.float)
    #print(f"action: {action}")
    #print(f"RETURNS log prob: {d.log_prob(action)}")
    model.saved_actions.append(SavedAction(d.log_prob(action).to(torch.float), state_value))

    # policy.saved_log_probs.append(d.log_prob(action))
    return action.detach().numpy()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).detach()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = np.float32( R - value.item())
        #log_prob = np.float32(log_prob)
        #print(f"adv: {advantage}")

        # calculate actor (policy) loss 
        policy_losses.append((-log_prob * advantage).to(torch.float))

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(torch.float)))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main(args, bandwidth=None):
    if bandwidth:
        env.bandwidth = bandwidth
    NUM_EPISODES = args.episodes
    running_reward = 10
    env.algorithm_name = "Actor Critic"
    print(f"Running {env.algorithm_name}")

    # run inifinitely many episodes
    for i_episode in range(NUM_EPISODES):

        # reset environment and episode reward
        state = env.reset_state()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, TIMESTEPS):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward = env.step(action, t)

            # if args.render:
            #     env.render()

            model.rewards.append(reward)
            ep_reward += reward


        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode != 0 and i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        
    return running_reward, env



if __name__ == '__main__':
    main(args)