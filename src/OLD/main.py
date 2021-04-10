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

def reset_state():
    operator1 = Operator("Operator 1")
    operator2 = Operator("Operator 2")
    operators = [operator1, operator2]
    allocator = Allocator(operators)
    return allocator, operators, allocator.get_state(operators)

def main():
    allocator, operators, state = reset_state()

    for t in range(0, TIMESTEPS):
        # Choose action to take given the state
        action = allocator.act(state)

        # Agent performs action
        next_state, reward = allocator.step(action, operators, t)

        # Remember 
        allocator.cache(state, next_state, action, reward)

        # Learn
        q, loss = allocator.learn(t)

        print(f'q: {q}, loss: {loss}')

        # Update state
        state = next_state

if __name__ == '__main__':
    main()