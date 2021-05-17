import numpy as np
RUNTIME = 60*60*24 # [s] 30 minutes
TIMESTEPS = 500 # Total t number of TIMESTEPS
T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep

AMPLITUDES = [20, 10]
PACKETS_PER_OPERATOR_PER_SECOND = 1580/60   
PACKET_SIZE = 8*10 # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size

SPECTRUM = np.arange(10, 40, 10)