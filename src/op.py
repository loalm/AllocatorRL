import numpy as np
import queue
from pprint import pprint
from constants import *
from packet import Packet

class Operator:
    lam = 2 # Average number of packets arriving every timestep.
    spectrum_size = 500
    block_size = 50
    def __init__(self, name):
        self.name = name
        self.cells = [] # TODO
        self.packet_distribution = np.random.poisson(lam=self.lam,size=TIMESTEPS)
                    # TODO: This should be varying more
        self.incoming_packets = [[Packet(step) for _ in range(self.packet_distribution[step])] 
                                    for step in range(TIMESTEPS)]

        self.packet_queue = queue.Queue()
        self.m = self.spectrum_size
        self.total_traffic_served = 0
        self.ema = np.zeros(TIMESTEPS) # Exponential moving avg of traffic

    def send_request(self):
        """
        TODO: How often should an operator be able to send a request to the
        allocator?
        NOTE: Suggestion:
        The operator may send a request to the allocator at most once
        per timestep and only when the operator does not have an ongoing
        spectrum loan.
        """
        pass

    def rr_schedule(self, t):
        """
        Schedules packets with Round Robin for the current timestep.
        Right now the cells are abstracted away.
        This could be changed to Proportional Fair scheduling in the future.
        This could also be done by an AI/RL algorithm.

        Return: The amount of traffic served (bits) during this timestep.
        """
        self.m = self.spectrum_size
        traffic_served = 0
        if t < len(self.incoming_packets):
            for p in self.incoming_packets[t]:
                self.packet_queue.put(p)

        pprint(self.packet_queue.queue)

        while self.m > 0 and not self.packet_queue.empty():
            print(f"a: {self.m}")
            p = self.packet_queue.get()
            chunk = p.spectral_efficiency * self.block_size # TODO: Double check this, only works assuming t=1s
            if self.m < chunk:
                return #Not enough resources a to schedule chunk
            
            p.size -= chunk
            traffic_served += chunk
            self.m -= self.block_size
            if p.size > 0:
                self.packet_queue.put(p)
            else:
                p.endtime = t
        self.calc_reward(traffic_served, t)
        return traffic_served

    def calc_reward(self, traffic_served, t):
        """
        Reward function (1): Total served traffic EMA
        # https://www.investopedia.com/terms/e/ema.asp
        # Don't know if accurate when t < n
        """
        value_t = traffic_served
        n = 20
        k = (2/(1+n))
        if t < TIMESTEPS:
            self.ema[t] = value_t * k + self.ema[t-1] * (1 - k)
        else:
            self.ema = np.append(self.ema, value_t * k + self.ema[t-1] * (1 - k))
        return self.ema[t]
