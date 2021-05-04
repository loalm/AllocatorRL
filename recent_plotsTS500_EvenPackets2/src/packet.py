import numpy as np

class Packet:
    size = 8 # 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
    id = 1 # Static variable. Initial package should have id 0.

    def __init__(self, arrival_time=None, id=None):
        self.set_id(id)
        self.arrival_time = arrival_time # [s]
        self.endtime = -1 # [s]
        self.set_efficiency() # ((bit/s)/Hz) NOTE: bit, not Byte

    def set_efficiency(self):
        mu, sigma = 10, 3 # mean and standard deviation
        self.spectral_efficiency = abs(np.random.normal(mu, sigma, 1)[0]) * 8 # Between 0-20 B/s/Hz with 99% probability

    def set_id(self, id):
        if id is None:
            self.id = Packet.id
            Packet.id += 1
        else:
            self.id = id
    def __repr__(self):
        return f"Packet #{self.id}"
    def __str__(self):
        return f"Packet #{self.id}"
