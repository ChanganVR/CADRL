import torch.nn as nn
from scipy.spatial import distance
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, fc_layers, reparametrization=True):
        super(ValueNetwork, self).__init__()
        self.reparametrization = reparametrization
        if reparametrization:
            state_dim = 15
        self.value_network = nn.Sequential(nn.Linear(state_dim, fc_layers[0]), nn.ReLU(),
                                           nn.Linear(fc_layers[0], fc_layers[1]), nn.ReLU(),
                                           nn.Linear(fc_layers[1], fc_layers[2]), nn.ReLU(),
                                           nn.Linear(fc_layers[2], 1))

    @staticmethod
    def rotate(state):
        # first translate the coordinate then rotate around the origin
        dx = state.pgx - state.px
        dy = state.pgy - state.py
        rot = np.arctan2(state.pgx-state.px, state.pgy-state.py)+np.pi

        dg = distance.euclidean((state.px, state.py), (state.pgx, state.pgy))
        v_pref = state.v_pref
        vx = state.vx * np.cos(rot) + state.vy * np.sin(rot)
        vy = state.vy * np.cos(rot) - state.vx * np.sin(rot)
        radius = state.radius
        theta = state.theta - rot
        vx1 = state.vx1 * np.cos(rot) + state.vy * np.sin(rot)
        vy1 = state.vy1 * np.cos(rot) - state.vx * np.sin(rot)
        px1 = (state.px1 + dx) * np.cos(rot) + (state.py1 + dy) * np.sin(rot)
        py1 = (state.py1 + dy) * np.cos(rot) - (state.px1 + dx) * np.sin(rot)
        radius1 = state.radius1
        radius_sum = radius + radius1
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        da = distance.euclidean((state.px, state.py), (state.px1, state.py1))

        new_state = (dg, v_pref, vx, vy, radius, theta, vx1, vy1, px1, py1,
                     radius1, radius_sum, cos_theta, sin_theta, da)
        return new_state

    def forward(self, state):
        if self.reparametrization:
            state = self.rotate(state)
        value = self.value_network(state)
        return value

