import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, fc_layers):
        super(ValueNetwork, self).__init__()
        self.value_network = nn.Sequential(nn.Linear(state_dim, fc_layers[0]), nn.ReLU(),
                                           nn.Linear(fc_layers[0], fc_layers[1]), nn.ReLU(),
                                           nn.Linear(fc_layers[1], fc_layers[2]), nn.ReLU(),
                                           nn.Linear(fc_layers[2], 1))

    def forward(self, state):
        value = self.value_network(state)
        return value

