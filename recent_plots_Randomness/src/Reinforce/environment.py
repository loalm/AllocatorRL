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


class Environment():
    def __init__(self, algorithm_name="NONAME"):
        self.algorithm_name = algorithm_name
        self.reset_state()
        self.bandwidth = 30
         # [Mhz] (Previous: 40)

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
        x = np.linspace(-np.pi, np.pi, RUNTIME)
        arrival_rates1 = (np.sin(x)*AMPLITUDES[0]).astype(int) + PACKETS_PER_OPERATOR_PER_SECOND#//60
        arrival_rates2 = (np.sin(x)*AMPLITUDES[1]).astype(int) + PACKETS_PER_OPERATOR_PER_SECOND#//60

        # Create the operators
        # start = time.time()
        self.operators = [Operator("Operator 1", arrival_rates1), 
                          Operator("Operator 2", arrival_rates2)]
        # end = time.time()
        # print("Time elasped: ", end-start)
        plot_packet_distribution(self, show_plot=True);exit()

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
            return math.log(traffic_sum)#traffic_sum 
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