import numpy as np
from src.constants import PACKET_SIZE
class Packet:
    size = PACKET_SIZE
    id = 1 # Static variable. Initial package should have id 0.

    def __init__(self, arrival_time=None, spectral_efficiency=None):
        # self.set_id(id)
        self.arrival_time = arrival_time # [s]
        self.spectral_efficiency = spectral_efficiency 
        self.endtime = -1 # [s]
        # self.set_efficiency() # ((bit/s)/Hz) NOTE: bit, not Byte

    # def set_efficiency(self):
    #     mu, sigma = 10, 3 # mean and standard deviation
    #     self.spectral_efficiency =  #max(0.5,abs(np.random.normal(mu, sigma, 1)[0]) * 8) # Between 0-20 B/s/Hz with 99% probability