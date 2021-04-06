import numpy as np

class Packet:
    size = 20 # Fixed packet size
    id = 1 # Static variable. Initial package should have id 0.

    def __init__(self, arrival_time, id=None):
        self.set_id(id)
        self.arrival_time = arrival_time # ?
        self.endtime = -1
        self.set_efficiency() # ((bit/s)/Hz)

    def set_efficiency(self):
        mu, sigma = 0.5, 0.5 # mean and standard deviation
        self.spectral_efficiency = abs(np.random.normal(mu, sigma, 1)[0])

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
