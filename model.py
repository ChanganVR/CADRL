import torch
import torch.nn as nn
import numpy as np


class IndexTranslator(object):
    def __init__(self, state):
        self.state = state
        self.px = self.state[:, 0].reshape(-1, 1)
        self.py = self.state[:, 1].reshape(-1, 1)
        self.vx = self.state[:, 2].reshape(-1, 1)
        self.vy = self.state[:, 3].reshape(-1, 1)
        self.radius = self.state[:, 4].reshape(-1, 1)
        self.pgx = self.state[:, 5].reshape(-1, 1)
        self.pgy = self.state[:, 6].reshape(-1, 1)
        self.v_pref = self.state[:, 7].reshape(-1, 1)
        self.theta = self.state[:, 8].reshape(-1, 1)
        self.px1 = self.state[:, 9].reshape(-1, 1)
        self.py1 = self.state[:, 10].reshape(-1, 1)
        self.vx1 = self.state[:, 11].reshape(-1, 1)
        self.vy1 = self.state[:, 12].reshape(-1, 1)
        self.radius1 = self.state[:, 13].reshape(-1, 1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, fc_layers, kinematic, reparametrization=True):
        super(ValueNetwork, self).__init__()
        self.reparametrization = reparametrization
        if reparametrization:
            state_dim = 15
        self.kinematic = kinematic
        self.value_network = nn.Sequential(nn.Linear(state_dim, fc_layers[0]), nn.ReLU(),
                                           nn.Linear(fc_layers[0], fc_layers[1]), nn.ReLU(),
                                           nn.Linear(fc_layers[1], fc_layers[2]), nn.ReLU(),
                                           nn.Linear(fc_layers[2], 1))

    def rotate(self, state, device):
        # first translate the coordinate then rotate around the origin
        # 'px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6         7        8       9      10     11    12       13
        state = IndexTranslator(state.cpu().numpy())
        dx = state.pgx - state.px
        dy = state.pgy - state.py
        rot = np.arctan2(state.pgy-state.py, state.pgx-state.px)

        dg = np.linalg.norm(np.concatenate([dx, dy], axis=1), axis=1, keepdims=True)
        v_pref = state.v_pref
        vx = state.vx * np.cos(rot) + state.vy * np.sin(rot)
        vy = state.vy * np.cos(rot) - state.vx * np.sin(rot)
        radius = state.radius
        if self.kinematic:
            theta = state.theta - rot
        else:
            theta = state.theta
        vx1 = state.vx1 * np.cos(rot) + state.vy1 * np.sin(rot)
        vy1 = state.vy1 * np.cos(rot) - state.vx1 * np.sin(rot)
        px1 = (state.px1 - state.px) * np.cos(rot) + (state.py1 - state.py) * np.sin(rot)
        py1 = (state.py1 - state.py) * np.cos(rot) - (state.px1 - state.px) * np.sin(rot)
        radius1 = state.radius1
        radius_sum = radius + radius1
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        da = np.linalg.norm(np.concatenate([state.px - state.px1, state.py - state.py1], axis=1), axis=1, keepdims=True)

        new_state = np.concatenate([dg, v_pref, vx, vy, radius, theta, vx1, vy1, px1, py1,
                                    radius1, radius_sum, cos_theta, sin_theta, da], axis=1)
        return torch.Tensor(new_state).to(device)

    def forward(self, state, device):
        if self.reparametrization:
            state = self.rotate(state, device)
        value = self.value_network(state)
        return value

