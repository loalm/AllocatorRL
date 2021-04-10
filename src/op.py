import numpy as np
import queue
from pprint import pprint
from src.constants import *
from src.packet import Packet
import math


class Operator:
    lam = 2 # Average number of packets arriving every timestep.
    bandwidth = 0
    #block_size = 5e6 # 5 MHz
    def __init__(self, name, packet_distribution = None):
        self.name = name
        self.cells = [] # TODO
        if packet_distribution is not None:
            self.packet_distribution = packet_distribution
        else:
            self.packet_distribution = np.random.poisson(lam=self.lam,size=TIMESTEPS)
                    # TODO: This should be varying more
        self.incoming_packets = [[Packet(step) for _ in range(self.packet_distribution[step])] 
                                    for step in range(TIMESTEPS)]

        self.packet_queue = queue.Queue()
        self.request = 0 # Request value to the allocator
        self.total_traffic_served = 0
        self.traffic_ema = np.zeros(TIMESTEPS+10000) # Exponential moving avg of traffic
        self.request_arr = []

    def schedule_packets(self, t):
        """
        Schedules packets for the current timestep.
        Right now the cells are abstracted away.
        This could be changed to Proportional Fair scheduling in the future.
        This could also be done by an AI/RL algorithm.

        Return: The amount of traffic served (Mb/s) during this timestep.
        """
        traffic_served = 0 # [bit / s]

        assert t < len(self.incoming_packets)
        
        for p in self.incoming_packets[t]:
            self.packet_queue.put(p)
        
        t_remaining = T_SLOT 

        while t_remaining > 0 and not self.packet_queue.empty():
            p = self.packet_queue.get()
            t_send_p = p.size / (self.bandwidth * p.spectral_efficiency) # Time required to send whole packet p [s]
            if t_send_p <= t_remaining:
                # Whole packet p is sent
                p.t2 = t*T_SLOT # Set the time t2 when p was fully sent.
                t_remaining -= t_send_p
                traffic_served += p.size
            else:
                p_chunk = self.bandwidth * p.spectral_efficiency * t_remaining # The chunk of packet p that can be sent [bits]
                p.size -= p_chunk
                self.packet_queue.put(p) # Cannot send whole packet p during timestep t.
                t_remaining = 0
                traffic_served += p_chunk
                break
        traffic_served /= T_SLOT # Divide by T_SLOT to get [bits / s]

        self.calc_reward(traffic_served, t)
        self.calc_request(t+1)
        return traffic_served
    
    def get_reward(self, t):
        """
        Reward function : Total served traffic EMA
        TODO: Add 5th percentile throughput minimum threshold
        [Mb / s]
        """
        return self.traffic_ema[t]

    def get_request(self):
        """
        Returns the resource request that is sent to the allocator.
        m = sum(s_j / e_j), where
        s_j : size of packet j
        e_j : spectral efficiency of packet j
        for every packet j in the packet queue
        [MHz * s]
        """
        return self.request


    def throughput():
        """
        Cell-edge throughput
        size / (t2-t1)
        ## Mb/s
        5% Worst percentile maximize 
        """
    
    def satisfied_traffic():
        """
        if 5% percentile satisified (> 1Mb/s):
            return traffic_served
        else
            return 0

            100Mhz Available Bandwidth
            0-20 B/s/Hz Spectral Efficiency
            1MB = 1*8Mb packet size
        """


    def calc_reward(self, traffic_served, t):
        """
        Calculates reward function : Total served traffic exponential moving average (EMA)
        # https://www.investopedia.com/terms/e/ema.asp
        # Don't know if accurate when t < n
        # NOTE: Could try a basic reward fn without EMA.
        [Mb / s]
        """
        #value_t = traffic_served
        #n = 5
        #k = (2/(1+n))
        #self.traffic_ema[t] = value_t * k + self.traffic_ema[t-1] * (1 - k)
        self.traffic_ema[t] = traffic_served

    def get_total_traffic_to_serve(self):
        tot_traffic = 0
        for t in range(TIMESTEPS):
            for p in self.incoming_packets[t]:
                tot_traffic += p.size
        return tot_traffic


    def calc_request(self, t):
        """
        Calculates the resource request that is sent to the allocator.
        request = sum(s_j / e_j) [MHz * s], where
        s_j : size of packet j 
        e_j : spectral efficiency of packet j
        for every packet j in the packet queue
        """
        t = t % TIMESTEPS

        self.request = 0
        #for tt in range(t, (t+10) % TIMESTEPS):
        #    self.request = sum([p.size / p.spectral_efficiency for p in self.incoming_packets[tt]])

        self.request = sum([p.size / p.spectral_efficiency for p in self.incoming_packets[t]])
        self.request += sum([p.size / p.spectral_efficiency for p in self.packet_queue.queue])

        self.request_arr.append(self.request)
        #print(f'Calc_request: {self.m}')

    # def send_request(self):
    #     """
    #     TODO: How often should an operator be able to send a request to the
    #     allocator?
    #     NOTE: Suggestion:
    #     The operator may send a request to the allocator at most once
    #     per timestep and only when the operator does not have an ongoing
    #     spectrum loan.
    #     """
    #     pass