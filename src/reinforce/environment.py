from src.op import Operator
from src.constants import TIMESTEPS
import matplotlib.pyplot as plot
import torch
import torch.nn.functional as F
import numpy as np
from typing import Iterator, List, Optional, Tuple, TypeVar
from src.Plotter.plotter import *
import math
from src.packet import Packet


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

        # q1 = list(o1.packet_queue.queue)[:10]
        # q2 = list(o2.packet_queue.queue)[:10]
        # packets_eff = [p.arrival_time / p.spectral_efficiency  for p in q1]
        # packets_eff.extend(0 for _ in range(10-len(q1)))
        # packets_eff.extend([p.size/p.spectral_efficiency for p in q1])
        # packets_eff.extend(0 for _ in range(10-len(q1)))
        
        # requests = np.array(packets_eff)

        requests = [
                    o1.request,
                    o2.request,
                    o1.five_percentile_throughput[t],
                    o2.five_percentile_throughput[t],
                    ]

        # print([o1.five_percentile_throughput[t], o2.five_percentile_throughput[t]])

        # if o1.five_percentile_throughput[t] == o2.five_percentile_throughput[t]:
        #     requests = [.5,.5]
        # elif o1.five_percentile_throughput[t] > o2.five_percentile_throughput[t]:
        #     requests = [0.4,0.7]
        # else:
        #     requests = [0.7,0.4]
        # requests = [o1.five_percentile_throughput[t], o2.five_percentile_throughput[t]]
        # print([o1.five_percentile_throughput[t], o2.five_percentile_throughput[t]])
        # requests.extend(torch.tensor([o1.request, o2.request]).float().tolist())
        # requests.extend(F.softmax(torch.tensor([o1.five_percentile_throughput[t],
        #                                         o2.five_percentile_throughput[t],]).float()).tolist())

        # print(requests)
        # print([o1.five_percentile_throughput[t], o2.five_percentile_throughput[t]])
        # requests.extend([o1.five_percentile_throughput[t],o2.five_percentile_throughput[t],t])
        state = np.array(requests)
        # state = np.append(state, len(self.operators[0].packet_queue.queue))
        # state = np.append(state, len(self.operators[1].packet_queue.queue))
        #print(f"state: {state}")
        return state

    def reset_state(self, n_operators: int = 2):
        # def generate_traffic_patterns(n_operators: int = 2, 
        #                               packet_means: List[int] = None,
        #                               packet_amplitudes: List[int] = None):
        #     x = np.linspace(-np.pi, np.pi, TIMESTEPS)
        #     packet_distributions = [(np.sin(x)*packet_amplitudes[i]
        #                             + packet_means[i]
        #                             + np.random.normal(0, 1, TIMESTEPS)).astype(int)
        #                             for i in range(n_operators)]

        #     return packet_distributions
        
        # packet_distributions = generate_traffic_patterns(packet_means=[20,20],
        #                                                  packet_amplitudes=[10,5])

        # self.operators = [Operator(f"Operator {i+1}", packet_distributions[i])
        #                   for i in range(n_operators)]
        x = np.linspace(-np.pi, np.pi, RUNTIME)
        arrival_rates1 = (np.sin(x)*70).astype(int) + 130
        arrival_rates2 = (np.sin(x+np.pi*3/4)*70).astype(int) + 130

       
        # Create the operators
        self.operators = [Operator("Operator 1", arrival_rates1), 
                          Operator("Operator 2", arrival_rates2)]
        # plot_packet_distribution(self, show_plot=True)

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
        #print(f"t: {t} five_percet_tps: {five_percentile_throughputs}")
        traffic_sum = sum([o.traffic_ema[t] for o in self.operators])
        if all(tp > 1 for tp in five_percentile_throughputs):
            return traffic_sum
        else:
            return 0#traffic_sum / 5# min(five_percentile_throughputs)
        # return 1/(1+abs(five_percentile_throughputs[0] - five_percentile_throughputs[1]))
        
        

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
            # reward += operator.get_reward(t)
            # reward += operator.get_quality_served_traffic(t)
            operator.bandwidth -= spectrum[i]

        reward = self.calc_quality_served_traffic(t)

        next_state = self.get_state(t)
        return next_state, reward