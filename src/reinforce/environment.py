from src.op import Operator
from src.constants import TIMESTEPS
import matplotlib.pyplot as plot
import torch
import torch.nn.functional as F
import numpy as np


class Environment():
    def __init__(self):
        self.reset_state()
        self.bandwidth = 40 # [Mhz] (Previous: 40)

    def get_state(self, t=0):
        """
        Returns the current state of the environment. 
        Currently the state is simply the requests from the operators.
        """
        [operator1, operator2] = self.operators
        requests = [self.operators[0].get_request(), self.operators[1].get_request()]
        #requests = F.softmax(torch.tensor(requests).float())
        state = np.array(requests)
        # state = np.append(state, len(self.operators[0].packet_queue.queue))
        # state = np.append(state, len(self.operators[1].packet_queue.queue))
        #print(f"state: {state}")
        return state

    def reset_state(self):
        # Create packet distribution for the two operators
        x = np.linspace(-np.pi, np.pi, TIMESTEPS)

        packet_dist1 = (np.sin(x)*10).astype(int) + 20
        packet_dist2 = (np.sin(x+np.pi*3/4)*10).astype(int) + 20

        for t in range(TIMESTEPS):
            packet_dist1[t] = np.random.poisson(lam=packet_dist1[t],size=1)
            packet_dist2[t] = np.random.poisson(lam=packet_dist2[t],size=1)

        #Create the operators
        self.operators = [Operator("Operator 1", packet_dist1), 
                          Operator("Operator 2", packet_dist2)]

        return self.get_state()

    def step(self, action, t):
        """
        Agent performs the action
        """
        def action_to_spectrum(a):
            s1 = action[0] * self.bandwidth
            s2 = action[1] * self.bandwidth
            return [s1, s2] 

        spectrum = action_to_spectrum(action)
        reward = 0
        for i, operator in enumerate(self.operators):
            operator.bandwidth += spectrum[i]
            operator.schedule_packets(t)
            #reward += operator.get_reward(t)
            reward += operator.get_quality_served_traffic(t)
            operator.bandwidth -= spectrum[i]

        next_state = self.get_state(t)
        return next_state, reward