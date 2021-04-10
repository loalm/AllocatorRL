#from src.allocator import Allocator
from src.op import Operator
from src.constants import TIMESTEPS
import matplotlib.pyplot as plot
import torch.nn.functional as F

import numpy as np


class Environment():
    def __init__(self):
        self.reset_state()
        self.bandwidth = 35
         # 100 Mhz

    def get_state(self, t=0):
        """
        Returns the current state of the environment. 
        """
        [operator1, operator2] = self.operators
        #state = [self.allocator.a, self.operators[0].get_request(), self.operators[1].get_request()]

        state = [self.operators[0].get_request(), self.operators[1].get_request()]
        #state = [1000,1000]
        return np.array(state)

    def reset_state(self):
        # Create the allocator        
        # Create packet distribution for the two operators
        x = np.linspace(-np.pi, np.pi, TIMESTEPS)
        y1 = (np.sin(x+np.pi*3/4)*10).astype(int) + 10

        packet_dist1 = (np.sin(x)*10).astype(int) + 10
        packet_dist2 = (np.sin(x+np.pi*3/4)*10).astype(int) + 10  

        for t in range(TIMESTEPS):
            packet_dist1[t] = np.random.poisson(lam=packet_dist1[t],size=1)
            packet_dist2[t] = np.random.poisson(lam=packet_dist2[t],size=1)

        #Create the operators
        self.operators = [Operator("Operator 1", packet_dist1), 
                          Operator("Operator 2", packet_dist2)]

        #self.plot_packets()

        return self.get_state()

    def step(self, action, t):
        """
        Agent performs the action
        """
        def action_to_spectrum(a):
            #print(f'Action: {a}')
            s1 = action[0] * self.bandwidth
            s2 = action[1] * self.bandwidth
            return [s1, s2] 

        spectrum = action_to_spectrum(action)
        reward = 0
        for i, operator in enumerate(self.operators):
            operator.bandwidth += spectrum[i]
            operator.schedule_packets(t)
            reward += operator.get_reward(t)
            operator.bandwidth -= spectrum[i]

        next_state = self.get_state(t)
        return next_state, reward

    
    def plot_packets(self):
        plot.plot(np.arange(TIMESTEPS), self.operators[0].packet_distribution, label = self.operators[0].name)
        plot.plot(np.arange(TIMESTEPS), self.operators[1].packet_distribution, label = self.operators[1].name)
        plot.legend()
        plot.title("Packet Distribution for Operator 1 and Operator 2")
        plot.xlabel("Timestep (t)")
        plot.ylabel("Number of Packets")
        plot.savefig("sin2.png")
        plot.show()