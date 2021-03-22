import numpy as np
import random
from queue import Queue
from constants import *

class Allocator:
    def __init__(self, operators):
        self.operators = operators
        self.spectrum_size = 100 # MHz
        self.block_size = 20 # MHz
        row = self.spectrum_size // self.block_size
        col = TIMESTEPS
        self.spectrum_pool = np.zeros((row, col))

        self.exploration_rate = 0.5
        self.a = self.spectrum_size

        self.memory = Queue()
        self.batch_size = 32
        self.action_dim = 1

    def allocate_spectrum():
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            action_idx = 0.5

        return action_idx

    def cache(self, state, next_state, action, reward):
        """
        Store the experience to self.memory (replay buffer)
        """
        self.memory.append((state, next_state, action, reward))

    def recall(self, experience):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        return batch
        

    def learn(self):
        pass

    def send_allocation(operator, allocation):
        pass