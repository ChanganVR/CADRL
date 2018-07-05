import math
import torch
from collections import namedtuple
from torch.utils.data import Dataset

FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])
JointState = namedtuple('JointState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta',
                                       'px1', 'py1', 'vx1', 'vy1', 'radius1'])
Velocity = namedtuple('Velocity', ['x', 'y'])
# v is velocity, under kinematic constraints, r is rotation angle otherwise it's speed direction
Action = namedtuple('Action', ['v', 'r'])


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)


class Trajectory(object):
    def __init__(self, gamma, goal_x, goal_y, radius, v_pref, times, positions, kinematic):
        self.gamma = gamma
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.radius = radius
        self.v_pref = v_pref
        self.times = times
        # time steps, 2 agents, xy coordinates
        self.positions = positions
        self.kinematic = kinematic

    @staticmethod
    def compute_value(gamma, time_to_goal, v_pref):
        return pow(gamma, time_to_goal * v_pref)

    def generate_state_value_pairs(self, device):
        positions = self.positions
        steps = positions.shape[0]
        pairs = list()
        for idx in range(1, steps):
            # calculate state of agent 0
            pos = positions[idx, 0, :]
            prev_pos = positions[idx - 1, 0, :]
            px, py = pos
            vx = pos[0] - prev_pos[0]
            vy = pos[1] - prev_pos[1]
            r = self.radius
            pgx = self.goal_x
            pgy = self.goal_y
            v_pref = self.v_pref
            if self.kinematic:
                theta = math.atan2(vy, vx)
            else:
                theta = 0

            # calculate state of agent 1
            pos1 = positions[idx, 1, :]
            prev_pos1 = positions[idx - 1, 1, :]
            px1, py1 = pos1
            vx1 = pos1[0] - prev_pos1[0]
            vy1 = pos1[1] - prev_pos1[1]
            r1 = self.radius

            state = torch.Tensor((px, py, vx, vy, r, pgx, pgy, v_pref, theta, px1, py1, vx1, vy1, r1)).to(device)
            value = torch.Tensor([self.compute_value(self.gamma, (self.times[-1] - self.times[idx]), self.v_pref)]).to(device)
            # value = torch.Tensor([1000])
            pairs.append((state, value))
        return pairs


