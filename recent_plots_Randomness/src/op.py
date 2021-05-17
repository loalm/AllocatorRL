import numpy as np
import queue
from collections import deque
from pprint import pprint
from src.constants import *
from src.packet import Packet
import math
import numpy as np


class Operator:
    lam = 2 # Average number of packets arriving every timestep.
    bandwidth = 0
    #block_size = 5e6 # 5 MHz
    def __init__(self, name, arrival_rates = None):
        self.name = name
        self.cells = [] # TODO
        self.packets_at_timestep = [[] for t in range(TIMESTEPS)]
        packets = []
        arrival_rates = np.random.poisson(lam=arrival_rates,size=RUNTIME)
        mu, sigma = 10, 3
        ses = []
        for s in range(RUNTIME):
            arrival_rate = arrival_rates[s]
            tt = np.linspace(0,1,arrival_rate)
            # packets.extend([Packet(arrival_time=s + tt[p]- 0.01)
            #                 for p in range(arrival_rate)])
            #np.random.rand()*0.01
            se = np.random.normal(mu, sigma, arrival_rate) * 8
            arrival_times = tt + s - 0.01 #+ np.random.rand(arrival_rate)*0.01
            packets.extend([Packet(arrival_time=arrival_times[p], 
                                    spectral_efficiency=se[p])
                            for p in range(arrival_rate)])

        for p in packets:
            timestep = int(p.arrival_time // T_SLOT)
            self.packets_at_timestep[timestep ].append(p)

        for t in range(TIMESTEPS):
            self.packets_at_timestep[t].sort(key=lambda p: p.arrival_time)

        self.packet_queue = deque()#queue.Queue()
        self.request = 0 # Request value to the allocator
        self.total_traffic_served = 0
        self.traffic_ema = np.zeros(TIMESTEPS+10000) # Exponential moving avg of traffic
        self.request_arr = []
        self.quality_served_traffic = np.zeros(TIMESTEPS)
        self.five_percentile_throughput = np.zeros(TIMESTEPS)
        self.utilisation = np.zeros(TIMESTEPS)
        self.throughput_arr = []

    def schedule_packets(self, t):
        """
        Schedules packets for the current timestep.
        Right now the cells are abstracted away.
        This could be changed to Proportional Fair scheduling in the future.
        This could also be done by an AI/RL algorithm.

        Return: The amount of traffic served (Mb/s) during this timestep.
        """
        traffic_served = 0 # [bit / s]

        assert t < len(self.packets_at_timestep)
        
        for p in self.packets_at_timestep[t]:
            self.packet_queue.appendleft(p)
        
        t_remaining = T_SLOT
        self.throughput_arr = []

        while t_remaining > 0 and self.packet_queue:
            p = self.packet_queue.pop()
            t_send_p = p.size / (self.bandwidth * p.spectral_efficiency) # Time required to send whole packet p [s]
            if t_send_p <= t_remaining:
                # Whole packet p is sent                    
                p.endtime = max(t*T_SLOT,p.arrival_time) + t_send_p # Set the time t2 when p was fully sent. [s]
                t_remaining -= t_send_p
                #print(f"Whole: {p.size}")
                if p.endtime - p.arrival_time < 1:
                    traffic_served += p.size
                    self.throughput_arr.append(p.size / (p.endtime - p.arrival_time)) # [Mb / s]
                else:
                    self.throughput_arr.append(0)
            else:
                p_chunk = self.bandwidth * p.spectral_efficiency * t_remaining # The chunk of packet p that can be sent [bits]
                #print(f"Chunk: {p_chunk}")
                p.size -= p_chunk
                chunk_endtime = max(t*T_SLOT,p.arrival_time) + t_send_p # Set the time t2 when p was fully sent. [s]
                if chunk_endtime - p.arrival_time < 1:
                    traffic_served += p_chunk
                    self.throughput_arr.append(p_chunk / (chunk_endtime - p.arrival_time)) # [Mb / s]
                    # Cannot send whole packet p during timestep t, add it to front of queue.
                    self.packet_queue.append(p) 
                    t_remaining = 0
                    break
                else: 
                    self.throughput_arr.append(0)

        self.utilisation[t] = (T_SLOT - t_remaining) / T_SLOT
        self.throughput_arr.extend([0 for p in self.packet_queue])

        traffic_served /= RUNTIME#T_SLOT # Divide by T_SLOT to get [bits / s]

        self.calc_quality_served_traffic(t)
        self.calc_reward(traffic_served, t)
        self.calc_request(t)
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

    def calc_quality_served_traffic(self, t):
        """
        if 5% percentile throughput (> 1Mb/s):
            return traffic_served
        else
            return 0

            100Mhz Available Bandwidth
            0-20 B/s/Hz Spectral Efficiency
            1MB = 1*8Mb packet size
        """
        if not self.throughput_arr:
            five_percentile_throughput = 0
        else:
            five_percentile_throughput = np.percentile(self.throughput_arr, 5)

        if five_percentile_throughput > 1:
            self.quality_served_traffic[t] = self.traffic_ema[t]
        else:
            self.quality_served_traffic[t] = 0#self.traffic_ema[t]/2 #0 #-100
        
        self.five_percentile_throughput[t] = five_percentile_throughput
    
    def get_quality_served_traffic(self, t):
        return self.quality_served_traffic[t]

    def calc_reward(self, traffic_served, t):
        """
        Calculates reward function : Total served traffic exponential moving average (EMA)
        # https://www.investopedia.com/terms/e/ema.asp
        # Don't know if accurate when t < n
        # NOTE: Could try a basic reward fn without EMA.
        [Mb / s]
        """
        self.traffic_ema[t] = traffic_served # / self.bandwidth

    def calc_request(self, t):
        """
        Calculates the resource request that is sent to the allocator.
        request = sum(s_j / e_j) [MHz * s], where
        s_j : size of packet j 
        e_j : spectral efficiency of packet j
        for every packet j in the packet queue
        """
        self.request = sum([p.size / (p.spectral_efficiency) for p in self.packet_queue])
 
        if t+1 != TIMESTEPS:
            self.request += sum([p.size / p.spectral_efficiency for p in self.packets_at_timestep[t+1]])

        # value_t = self.request
        # n = min(t,5)
        # k = (2/(1+n))
        # if t == 0:
        #     self.request = value_t
        # else:
        #     self.request = value_t * k + self.request_arr[(t-1)] * (1 - k)
        self.request_arr.append(self.request)
