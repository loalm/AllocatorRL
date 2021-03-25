from torch import nn
import copy

class AllocateNet(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(AllocateNet, self).__init__()

        self.fc1_dims = 5

        self.online = nn.Sequential(
            nn.Linear(*input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, n_actions),
            nn.Softmax()
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, model):
        if model == 'online':
            return self.online(state)
        elif model == 'target':
            return self.target(state)