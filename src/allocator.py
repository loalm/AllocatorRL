import torch
import numpy as np
import random
from queue import Queue
from constants import *
from neural import AllocateNet

class Allocator:
    def __init__(self, operators):
        self.operators = operators
        self.spectrum_size = 100 # MHz
        self.block_size = 20 # MHz
        row = self.spectrum_size // self.block_size
        col = TIMESTEPS
        self.spectrum_pool = np.zeros((row, col))

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.a = self.spectrum_size

        self.min_learn = 40 # min. experiences before training
        self.learn_every = 3 # no. of experiences between updates to Q_online
        self.sync_every = 10 # no. of experiences between Q_target & Q_online sync

        self.save_every = 1000 # no. of experiences between saving Mario net

        self.memory = Queue()
        self.batch_size = 32
        self.state_dim = (3,1)
        self.action_dim = 5

        self.net = AllocateNet(self.state_dim, self.action_dim).float()

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.net = self.net.to(device = 'cuda')
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss() #self.loss = torch.nn.MSELoss()

    def get_state(self, operators):
        [operator1, operator2] = operators
        state = (self.a, operator1.get_request(), operator2.get_request())
        return state

    def allocate_spectrum():
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action
        
        Input:
        state: A single observation of the current state, dimension is (state_dim)
        Output:
        action_idx (int) : An integer representing which action Mario will perform.
        """     

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.Floattensor(state).cuda() if self.use_cuda else torch.Floattensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model = 'online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return action_idx

    def step(self, action_idx, operators, t):
        """
        Agent performs the action
        """
        def action_to_spectrum(action_idx):
            """
                Converts the action from the discrete action space to a continuous spectrum allocation
                for operator 1 and operator 2.  
            """
            p1 = [1, 0.75, 0.5, 0.25, 0] # Percent of spectrum allocated to operator 1
            p1 = p1[action_idx]
            p2 = 1 - p1  # Percent of spectrum allocated to operator 2
            s1 = p1 * self.spectrum_size 
            s2 = p2 * self.spectrum_size
            return [s1, s2] 

        s = action_to_spectrum(action_idx)

        reward = 0
        for i, op in enumerate(operators):
            op.spectrum_size += s[i]
            op.rr_schedule(t)
            reward += op.get_reward(t)
            op.spectrum_size -= s[i]

        next_state = self.get_state(operators)
        return next_state, reward

    def cache(self, state, next_state, action, reward):
        """
        Store the experience to self.memory (replay buffer)
        """

        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])

        self.memory.put( (state, next_state, action, reward) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        print(f"queue size: {len(self.memory.queue)}, batch size: {self.batch_size}")
        batch = random.sample(self.memory.queue, self.batch_size)
        print(f"batch : {batch}")
        state, next_state, action, reward = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ] # Q_online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target',)[
            np.arange(0, self.batch_size),
            best_action
        ]
        return (reward + self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self, t):
        if t % self.sync_every == 0:
            self.sync_Q_target()

        #if t % self.save_every == 0:
        #    self.save()

        if t < self.min_learn:
            return None, None

        if t % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def send_allocation(operator, allocation):
        pass

