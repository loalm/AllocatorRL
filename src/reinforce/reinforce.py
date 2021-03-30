import Environment

env = Environment()

class Policy(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

def select_action(state):
    pass

def finish_episode():
    pass

def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(100):
            action = select_action(state)
            state, reward = env.step(action)
            


if __name__ == '__main__':
    main()



