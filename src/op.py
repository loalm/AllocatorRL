import numpy as np
import queue
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
        for s in range(RUNTIME):
            arrival_rate = arrival_rates[s]#np.random.poisson(lam=arrival_rates[s],size=1)[0]
            tt = np.linspace(0,1,arrival_rate)
            packets.extend([Packet(arrival_time=s + tt[p]- 0.01)
                             for p in range(arrival_rate)])

        for p in packets:
            timestep = int(p.arrival_time // T_SLOT)
            self.packets_at_timestep[timestep].append(p)

        for t in range(TIMESTEPS):
            self.packets_at_timestep[t].sort(key=lambda p: p.arrival_time)

        # if packets_at_timestep is not None:
        #     self.packets_at_timestep = packets_at_timestep
        # else:
        #     self.packets_at_timestep = np.random.poisson(lam=self.lam,size=TIMESTEPS)
        #             # TODO: This should be varying more
        # self.packets_at_timestep = [[Packet(arrival_time=step*T_SLOT + np.random.rand()*T_SLOT) for _ in range(self.packets_at_timestep[step])] 
        #                             for step in range(TIMESTEPS)]

        self.packet_queue = queue.Queue()
        self.request = 0 # Request value to the allocator
        self.total_traffic_served = 0
        self.traffic_ema = np.zeros(TIMESTEPS+10000) # Exponential moving avg of traffic
        self.request_arr = []
        self.quality_served_traffic = np.zeros(TIMESTEPS)
        self.five_percentile_throughput = np.zeros(TIMESTEPS)
        self.utilisation = np.zeros(TIMESTEPS)

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
            self.packet_queue.put(p)
        
        t_remaining = T_SLOT
        self.throughput_arr = []

        while t_remaining > 0 and not self.packet_queue.empty():
            p = self.packet_queue.get()
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
                    self.packet_queue.put(p) # Cannot send whole packet p during timestep t.
                    t_remaining = 0
                    break
                else: 
                    self.throughput_arr.append(0)

        self.utilisation[t] = (T_SLOT - t_remaining) / T_SLOT
        self.throughput_arr.extend([0 for p in self.packet_queue.queue])

        traffic_served /= T_SLOT # Divide by T_SLOT to get [bits / s]

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
        # print(f"t: {t}", self.throughput_arr)
        # print(f"t: {t} {self.name} TP Arr: {self.throughput_arr}")
        if not self.throughput_arr:
            five_percentile_throughput = 0
        else:
            five_percentile_throughput = np.percentile(self.throughput_arr, 5)

        #print(f"t: {t} {self.name} 5TP: {five_percentile_throughput}")
        if five_percentile_throughput > 1:
            self.quality_served_traffic[t] = self.traffic_ema[t]
        else:
            self.quality_served_traffic[t] = self.traffic_ema[t]/2 #0 #-100
        
        self.five_percentile_throughput[t] = five_percentile_throughput
    
    def get_quality_served_traffic(self, t):
        return self.quality_served_traffic[t]


    # if throughtput > threshold:
        #reward = 1/Spectrum
    # else:
        #reward = 0


    def calc_reward(self, traffic_served, t):
        """
        Calculates reward function : Total served traffic exponential moving average (EMA)
        # https://www.investopedia.com/terms/e/ema.asp
        # Don't know if accurate when t < n
        # NOTE: Could try a basic reward fn without EMA.
        [Mb / s]
        """
        # value_t = traffic_served
        # n = min(t,50)
        # k = (2/(1+n))
        # self.traffic_ema[t] = value_t * k + self.traffic_ema[t-1] * (1 - k)
        self.traffic_ema[t] = traffic_served # / self.bandwidth

    def get_total_traffic_to_serve(self):
        tot_traffic = 0
        for t in range(TIMESTEPS):
            for p in self.packets_at_timestep[t]: 
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

        self.request = 0
        # for tt in range(t, (t+10) % TIMESTEPS):
        #    self.request = sum([p.size / p.spectral_efficiency for p in self.packets_at_timestep[tt]])
        
        # if self.five_percentile_throughput[t] < 1:
            # self.request = 100_000
        # else:
        self.request = sum([p.size / (p.spectral_efficiency) for p in self.packet_queue.queue])
    

        if t+1 != TIMESTEPS:
            self.request += sum([p.size / p.spectral_efficiency for p in self.packets_at_timestep[t+1]])
        # if self.five_percentile_throughput[t] < 1:
            # self.request += sum([p.size / (p.spectral_efficiency) for p in self.packet_queue.queue])
            # self.request = 0.5
        # self.request = 0.5

        # if 0 < t and t < 300:
        #     print(f't: {t} op: {self.name} tp: {self.five_percentile_throughput[t]}')
        # self.request = 1 / (1+self.five_percentile_throughput[t])
        self.request_arr.append(self.request)
        # if self.five_percentile_throughput[t] < 1:
        #     print("BAD!")
        # print(f't: {t} {self.name}  tp : {self.five_percentile_throughput[t]}')