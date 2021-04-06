import numpy as np
import queue
from pprint import pprint
from src.constants import *
from src.packet import Packet

class Operator:
    lam = 2 # Average number of packets arriving every timestep.
    spectrum_size = 500
    block_size = 15
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

    def rr_schedule(self, t):
        """
        Schedules packets with Round Robin for the current timestep.
        Right now the cells are abstracted away.
        This could be changed to Proportional Fair scheduling in the future.
        This could also be done by an AI/RL algorithm.

        Return: The amount of traffic served (bits) during this timestep.
        """
        #print(f"rr_schedule {self.name} Spectrum size: {self.spectrum_size}")
        current_spectrum = self.spectrum_size
        traffic_served = 0
        if t < len(self.incoming_packets):
            for p in self.incoming_packets[t]:
                self.packet_queue.put(p)

        #pprint(self.packet_queue.queue)

        while current_spectrum > 0 and not self.packet_queue.empty():
            #print(f"a: {self.m}")
            p = self.packet_queue.get()
            chunk = p.spectral_efficiency * self.block_size # TODO: Double check this, only works assuming t=1s
            if current_spectrum < chunk:
                break # Not enough resources to schedule chunk
            
            #print(f"p.size: {p.size}, chunk: {chunk}, p.se: {p.spectral_efficiency}")
            traffic_served += min(p.size, chunk)
            p.size -= chunk
            current_spectrum -= self.block_size
            if p.size > 0:
                self.packet_queue.put(p)
            else:
                p.endtime = t # Packet done
        self.calc_reward(traffic_served, t)
        self.calc_request()
        return traffic_served
    
    def get_reward(self, t):
        """
        Reward function : Total served traffic EMA
        TODO: Add 5th percentile throughput minimum threshold
        """
        return self.traffic_ema[t]

    def get_request(self):
        """
        Returns the resource request that is sent to the allocator.
        m = sum(s_j / e_j), where
        s_j : size of packet j
        e_j : spectral efficiency of packet j
        for every packet j in the packet queue
        """
        return self.request

    def calc_reward(self, traffic_served, t):
        """
        Calculates reward function : Total served traffic exponential moving average (EMA)
        # https://www.investopedia.com/terms/e/ema.asp
        # Don't know if accurate when t < n
        # NOTE: Could try a basic reward fn without EMA.
        """
        value_t = traffic_served
        n = 20
        k = (2/(1+n))
        #self.traffic_ema[t] = value_t * k + self.traffic_ema[t-1] * (1 - k)
        self.traffic_ema[t] = traffic_served

    def get_total_traffic_to_serve(self):
        tot_traffic = 0
        for t in range(TIMESTEPS):
            for p in self.incoming_packets[t]:
                tot_traffic += p.size
        return tot_traffic


    def calc_request(self):
        """
        Calculates the resource request that is sent to the allocator.
        m = sum(s_j / e_j), where
        s_j : size of packet j
        e_j : spectral efficiency of packet j
        for every packet j in the packet queue
        """
        self.request = sum([p.size / p.spectral_efficiency for p in self.packet_queue.queue])
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