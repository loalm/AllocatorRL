from src.op import Operator
from src.constants import *
import matplotlib.pyplot as plot
import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterator, List, Optional, Tuple, TypeVar
from src.Plotter.plotter import *
import math
from src.packet import Packet
import time

eps = np.finfo(np.float32).eps.item()

class Environment():
    def __init__(self, algorithm_name="NONAME"):
        self.algorithm_name = algorithm_name
        self.bandwidth = 30 # [Mhz] (Previous: 40)
        self.reset_state()

    def get_state(self, t=0):
        """
        Returns the current state of the environment. 
        Currently the state is simply the requests from the operators.
        """
        [o1, o2] = self.operators
        requests = [
                    o1.request,
                    o2.request,
                    o1.five_percentile_throughput[t],
                    o2.five_percentile_throughput[t],
        ]
        state = np.array(requests)
        return state

    def reset_state(self, n_operators: int = 2):
        x_t = np.linspace(-np.pi, np.pi, TIMESTEPS)

        self.operators = []
        for i in range(n_operators):
            PACKETS_PER_OPERATOR_PER_TIMESTEP = [p * RUNTIME // TIMESTEPS for p in PACKETS_PER_OPERATOR_PER_SECOND]
            PACKET_AMPLITUDES_PER_TIMESTEP =  [p * RUNTIME // TIMESTEPS for p in PACKET_AMPLITUDES_PER_SECOND]

            arrival_rates = (np.sin(x_t)*PACKET_AMPLITUDES_PER_TIMESTEP[i] + PACKETS_PER_OPERATOR_PER_TIMESTEP[i]).astype(int)
            # plot.plot(distribution)
            # plot.show()
            self.operators.append(Operator(f"Operator {i}", arrival_rates))
        # plot_packet_distribution(self, hourly = False); quit()
        return self.get_state()

    def calc_quality_served_traffic(self, t):
        """
        Returns total traffic served during timestep t if ALL OPERATORS 5% throughput is > 1Mb/s

        if ALL OPERATORS 5% percentile throughput (> 1Mb/s):
            return traffic_served
        else
            return 0
        """
        five_percentile_throughputs = [np.percentile(o.throughput_arr, 5) for o in self.operators]
        traffic_sum = sum([o.traffic_ema[t] for o in self.operators])
        if all(tp > 1 for tp in five_percentile_throughputs):
            return traffic_sum 
        else:
            return 0
        
        

    def step(self, action, t):
        """
        Agent performs the action
        """
        def action_to_spectrum(a):
            s1 = action[0] * self.bandwidth
            s2 = action[1] * self.bandwidth
            return [s1, s2] 

        spectrum = action_to_spectrum(action)
        for i, operator in enumerate(self.operators):
            operator.bandwidth += spectrum[i]
            operator.schedule_packets(t)
            operator.bandwidth -= spectrum[i]

        reward = self.calc_quality_served_traffic(t)

        next_state = self.get_state(t)
        return next_state, reward