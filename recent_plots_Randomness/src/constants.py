import numpy as np

####### Downscaled 1 minute: USE THIS TO TRAIN! #####
# x = 1
# RUNTIME = 60*x # [s] 30 minutes
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*2//x, 100*2//x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2//x, 1580*2//x]#1580/60   
# PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(145, 1010, 10)#87.5
####### #######

####### Downscaled 2h Checkpoint: Outperforms baseline! #####
# x = 60*2
# RUNTIME = 60*x # [s] 30 minutes
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*2//x, 100*2//x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2//x, 1580*2//x]#1580/60   
# PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(140, 1010, 10)#87.5
####### #######

####### Downscaled 5h Checkpoint: Outperforms baseline #####
# x = 60*5
# RUNTIME = 60*x # [s] 30 minutes
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*2//x, 100*2//x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2//x, 1580*2//x]#1580/60   
# PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(140, 1010, 10)#87.5
####### #######

####### Downscaled 10h Checkpoint: Outperforms baseline#####
###### #######
# x = 60*10
# RUNTIME = 60*x # [s] 30 minutes
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*4//x, 100*4//x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*4//x, 1580*4//x]#1580/60   
# PACKET_SIZE = 8/2*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(140, 1010, 10)#87.5
####### #######

####### 24h Checkpoint: Outperforms baseline with ~30% #####
####### #######
x = 60*24
RUNTIME = 60*x # [s] 30 minutes
TIMESTEPS = 500 # Total t number of TIMESTEPS
T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
LEARNING_RATE = 0.0005
PACKET_AMPLITUDES_PER_SECOND = [300*8//x, 100*8//x]
PACKETS_PER_OPERATOR_PER_SECOND = [1580*4//x, 1580*4//x]#1580/60   
PACKET_SIZE = 8/2*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
NUM_CELLS = 5
SPECTRUM = np.arange(135, 1010, 10)#87.5
####### #######


print(f"Average Packets / sec: {(RUNTIME/TIMESTEPS) * PACKETS_PER_OPERATOR_PER_SECOND[0]}")


####### Upscaled #####
####### #######
# RUNTIME = 60*60 # [s] 30 minutes
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# AMPLITUDES = [300, 200]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580, 1000]#1580/60   
# PACKET_SIZE = 8 # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(90, 110, 10)#87.5


# RUNTIME = 60*60*24 # [s] 1 day 
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep

# AMPLITUDES = [20, 10]
# PACKETS_PER_OPERATOR_PER_SECOND = [50, 25]#1580/60   
# PACKET_SIZE = 8*20 # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size

# SPECTRUM = np.arange(50, 4000, 10)