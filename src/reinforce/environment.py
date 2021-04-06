from src.allocator import Allocator
from src.op import Operator
from src.constants import TIMESTEPS

import numpy as np


class Environment():
    def __init__(self):
        self.reset_state()

    def get_state(self):
        """
        Returns the current state of the environment. 
        """
        [operator1, operator2] = self.operators
        state = [self.allocator.a, self.operators[0].get_request(), self.operators[1].get_request()]
        return np.array(state)

    def reset_state(self):
        # Create the allocator
        self.allocator = Allocator()
        
        # Create packet distribution for the two operators
        x = np.linspace(-np.pi, np.pi, TIMESTEPS)
        y1 = (np.sin(x+np.pi*3/4)*10).astype(int) + 10
        packet_dist1 = (np.sin(x)*10).astype(int) + 10
        packet_dist2 = (np.sin(x+np.pi*3/4)*10).astype(int) + 10  

        # plot.plot(np.arange(TIMESTEPS), packet_dist1, label = "Operator 1")
        # plot.plot(np.arange(TIMESTEPS),packet_dist1, label = "Operator 2")
        # plot.legend()
        # plot.title("Packet Distribution for Operator 1 and Operator 2")
        # plot.xlabel("Timestep (t)")
        # plot.ylabel("Number of Packets")
        # plot.savefig("sin2.png")
        # plot.show()      
        
        #Create the operators
        self.operators = [Operator("Operator 1", packet_dist1), 
                          Operator("Operator 2", packet_dist2)]
        return self.get_state()

    def step(self, action, t):
        """
        Agent performs the action
        """
        def action_to_spectrum(a):
            #print(f'Action: {a}')
            s1 = action[0] * self.allocator.spectrum_size
            s2 = action[1] * self.allocator.spectrum_size
            return [s1, s2] 

        s = action_to_spectrum(action)
        reward = 0
        for i, op in enumerate(self.operators):
            op.spectrum_size += s[i]
            op.rr_schedule(t)
            reward += op.get_reward(t)
            op.spectrum_size -= s[i]

        next_state = self.get_state()
        return next_state, reward
