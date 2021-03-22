import numpy as np
from pprint import pprint
from constants import *
from op import Operator
from allocator import Allocator

#TODO: Create a baseline static spectrum allocator

class User:
    pass # Abstracted away for now?

class Cell:
    pass # Abstracted away for now?

def reset_state(allocator, operators):
    operator1 = Operator("Operator 1")
    operator2 = Operator("Operator 2")
    operators = [operator1, operator2]
    allocator = Allocator(operators)

    state = (allocator.a, operator1.m, operator2.m)
    return state


def main():
    operator1 = Operator("Operator 1")
    operator2 = Operator("Operator 2")
    operators = [operator1, operator2]
    allocator = Allocator(operators)
    pprint(operator1.incoming_packets)
    t = 0
    while t < TIMESTEPS or (not operator1.packet_queue.empty()):
        operator1.rr_schedule(t)
        t += 1

    state = reset_state(allocator, operators)

    while True:
        # Choose action to take given the state
        action = allocator.act(state)

        # Agent performs action
        next_state, reward = allocator.step(action)

        # Remember 
        allocator.cache(state, next_state, action, reward)

        # Learn
        q, loss = allocator.learn()

        # Update state
        state = next_state

    print(operator1.ema)

if __name__ == '__main__':
    main()