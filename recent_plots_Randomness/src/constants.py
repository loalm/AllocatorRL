import numpy as np

####### Downscaled 1 minute: USE THIS TO TRAIN! #####
x = 60*24
RUNTIME = 60*x # [s] 
TIMESTEPS = 500 # Total t number of TIMESTEPS
T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
LEARNING_RATE = 0.0005# NOTE
PACKET_AMPLITUDES_PER_SECOND = [300*2/x, 200*2/x]
PACKETS_PER_OPERATOR_PER_SECOND = [1580*2/x, 1000*2/x]#1580/60   
PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
NUM_CELLS = 10
SPECTRUM = np.arange(65, 10010, 5)#check: 140, 160, 180, 200, 220, 240, 260 with MAX
####### #######

####### Downscaled 24h Checkpoint: Outperforms baseline #####
# x = 60*24 # Scaling factor
# RUNTIME = 60*x # [s] 
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [500*2/x, 100*2/x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2/x, 1580*2/x]  
# PACKET_SIZE = 8*x # [Mb]
# NUM_CELLS = 10
# SPECTRUM = np.arange(60, 120, 5)
####### #######

####### Downscaled 2h Checkpoint: Outperforms baseline! #####
# x = 60*2
# RUNTIME = 60*x # [s] 
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*1/x, 100*1/x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*1//x, 1580*1//x]#1580/60   
# PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(90, 1010, 10)#87.5
####### #######

####### Downscaled 5h Checkpoint: Outperforms baseline #####
# x = 60*5
# RUNTIME = 60*x # [s] 
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [300*2//x, 100*2//x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2//x, 1580*2//x]#1580/60   
# PACKET_SIZE = 8*x # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 5
# SPECTRUM = np.arange(145, 1010, 10)#87.5
####### #######

####### Downscaled 10h Checkpoint: Outperforms baseline#####
###### #######
# x = 60*10
# RUNTIME = 60*x # [s] 
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
# x = 60*24
# RUNTIME = 60*x # [s] 
# TIMESTEPS = 500 # Total t number of TIMESTEPS
# T_SLOT = RUNTIME/TIMESTEPS # [s] The length of 1 timestep
# LEARNING_RATE = 0.0005
# PACKET_AMPLITUDES_PER_SECOND = [500*2/x, 100*2/x]
# PACKETS_PER_OPERATOR_PER_SECOND = [1580*2/x, 1580*2/x]#1580/60   
# PACKET_SIZE = 8 # [Mb] 8*10^6 bits = 1 MB = 1*8 Mb fixed packet size
# NUM_CELLS = 1
# SPECTRUM = np.arange(10, 1010, 10)#87.5
####### #######


print(f"Average Packets / Second: {PACKETS_PER_OPERATOR_PER_SECOND[0]}")
print(f"Average Packets / Timestep: {(RUNTIME/TIMESTEPS) * PACKETS_PER_OPERATOR_PER_SECOND[0]}")
