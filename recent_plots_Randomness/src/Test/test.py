from src.op import Operator
from src.constants import *
import matplotlib.pyplot as plot
import numpy as np
from src.constants import *


def main():
    x = np.linspace(-np.pi, np.pi, TIMESTEPS)
    y1 = (np.sin(x+np.pi*3/4)*10).astype(int) + 10
    print(f'packet_distribution: {y1}')

    op1 = Operator("Op1", packet_distribution = y1)
    op1.bandwidth = 500
    op1.block_size = 15
    tot_traffic = op1.get_total_traffic_to_serve()
    tot_reward = 0
    for t in range(TIMESTEPS):
        op1.schedule_packets(t)
        reward = op1.get_reward(t)
        tot_reward += reward
        #print(f't: {t} Reward: {reward}')
        #print(f'queue: {op1.packet_queue.queue}')
    
    print(f'Total traffic to serve: {tot_traffic}')
    print(f'Total reward: {tot_reward}')

    # plot.plot(op1.packet_distribution)
    # #plot.plot(op1.packet_distribution)
    # plot.title('Packet distribution')
    # plot.xlabel('Timestep (t)')
    # plot.ylabel('Number of packets')
    # plot.savefig('packet_distribution.png')
    # plot.show()
    # plot.close()

def plott():
    print("Plotting...")
    # Get x values of the sine wave
    x = np.linspace(-np.pi, np.pi, TIMESTEPS)
    # Amplitude of the sine wave is sine of a variable like time
    y1 = np.sin(x)*10 + 10
    y2 = np.sin(x+np.pi*3/4)*10 + 10

    # Plot a sine wave using time and amplitude obtained for the sine wave
    plot.plot(np.arange(TIMESTEPS), y1, label = "Operator 1")
    plot.plot(np.arange(TIMESTEPS), y2, label = "Operator 2")
    plot.title("Packet Distribution for Operator 1 and Operator 2")
    plot.xlabel("Timestep (t)")
    plot.ylabel("Number of packets")
    plot.legend()
    plot.savefig("sin1.png")
    plot.show()


if __name__ == '__main__':
    main()