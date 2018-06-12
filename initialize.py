import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import os
import math
import logging


def compute_value(gamma, time_to_goal, v_pref):
    return pow(gamma, time_to_goal * v_pref)


class Trajectory(object):
    def __init__(self, gamma, goal_x, goal_y, radius, v_pref, times, positions, kinematic_constrained):
        self.gamma = gamma
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.radius = radius
        self.v_pref = v_pref
        self.times = times
        # time steps, 2 agents, xy coordinates
        self.positions = positions
        self.kinematic_constrained = kinematic_constrained

    def generate_state_value_pairs(self):
        positions = self.positions
        steps = positions.shape[0]
        pairs = list()
        for idx in range(1, steps - 1):
            # calculate state of agent 0
            pos = positions[idx, 0, :]
            prev_pos = positions[idx - 1, 0, :]
            next_pos = positions[idx + 1, 0, :]
            px, py = pos
            vx = pos[0] - prev_pos[0]
            vy = pos[1] - prev_pos[1]
            r = self.radius
            pgx = self.goal_x
            pgy = self.goal_y
            v_pref = self.v_pref
            if self.kinematic_constrained:
                theta = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
            else:
                theta = 0

            # calculate state of agent 1
            pos1 = positions[idx, 1, :]
            prev_pos1 = positions[idx - 1, 1, :]
            px1, py1 = pos1
            vx1 = pos1[0] - prev_pos1[0]
            vy1 = pos1[1] - prev_pos1[1]
            r1 = self.radius

            state = torch.Tensor((px, py, vx, vy, r, pgx, pgy, v_pref, theta, px1, py1, vx1, vy1, r1))
            value = torch.Tensor([compute_value(self.gamma, (self.times[-1] - self.times[idx]), self.v_pref)])
            # value = torch.Tensor([1000])
            pairs.append((state, value))
        return pairs


class StateValue(Dataset):
    def __init__(self, state_value_pairs):
        super(StateValue, self).__init__()
        self.state_value_pairs = state_value_pairs

    def __len__(self):
        return len(self.state_value_pairs)

    def __getitem__(self, item):
        return self.state_value_pairs[item]


def load_data(traj_dir, gamma, kinematic_constrained):
    state_value_pairs = list()
    for traj_file in os.listdir(traj_dir):
        # parse trajectory data to state-value pairs
        with open(os.path.join(traj_dir, traj_file)) as fo:
            lines = fo.readlines()
            times = list()
            positions = list()
            for line in lines[2:]:
                line = line.split()
                times.append(float(line[0]))
                position = [[float(x) for x in re.sub('[()]', '', po).split(',')] for po in line[1:]]
                positions.append(position)
            positions = np.array(positions)

        trajectory1 = Trajectory(gamma, *[float(x) for x in lines[0].split()],
                                 times, positions, kinematic_constrained)
        trajectory2 = Trajectory(gamma, *[float(x) for x in lines[1].split()],
                                 times, positions[:, ::-1, :], kinematic_constrained)
        state_value_pairs += trajectory1.generate_state_value_pairs() + trajectory2.generate_state_value_pairs()

    logging.info('Total number of state_value pairs: {}'.format(len(state_value_pairs)))
    state_value_dataset = StateValue(state_value_pairs)

    return state_value_dataset


def initialize(model, model_config, env_config):
    gamma = model_config.getfloat('model', 'gamma')
    traj_dir = model_config.get('init', 'traj_dir')
    num_epochs = model_config.getint('init', 'num_epochs')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')
    kinematic_constrained = env_config.getboolean('agent', 'kinematic_constrained')

    state_value_dataset = load_data(traj_dir, gamma, kinematic_constrained)
    data_loader = DataLoader(state_value_dataset, batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    for epoch in range(num_epochs):
        epoch_loss = 0
        lr_scheduler.step()
        for data in data_loader:
            inputs, values = data
            inputs = Variable(inputs)
            values = Variable(values)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item()

        logging.info('Loss in epoch {}: {:.2f}'.format(epoch, epoch_loss))
    return model
