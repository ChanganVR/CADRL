import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import math
import sys
import logging
import random
import itertools
import argparse
import configparser
from collections import defaultdict
from model import ValueNetwork
from env import *
from initialize import initialize, compute_value


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * self.capacity
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return sum([x is not None for x in self.memory])


def filter_velocity(joint_state, state_sequence, agent_idx):
    """
    Compute the other agent's average velocity in last two time steps

    """
    if agent_idx not in state_sequence:
        prev_v = Velocity(0, 0)
    else:
        prev_v = Velocity(state_sequence[agent_idx][-1].vx1, state_sequence[agent_idx][-1].vy1)
    current_v = Velocity(joint_state.vx1, joint_state.vy1)
    filtered_v = Velocity((prev_v.x+current_v.x)/2, (prev_v.y+current_v.y)/2)

    return filtered_v


def propagate(state, v_est, delta_t=1):
    """
    Compute approximate next state with estimated velocity/action

    """
    if isinstance(state, ObservableState) and isinstance(v_est, Velocity):
        # propagate state of the other agent
        new_px = state.px + v_est.x * delta_t
        new_py = state.py + v_est.y * delta_t
        # TODO: how to propagate
        state = ObservableState(new_px, new_py, v_est.x, v_est.y, state.radius)
    elif isinstance(state, FullState) and isinstance(v_est, Action):
        # propagate state of current agent
        # perform action without rotation TODO: impose kinematic constraint and theta
        new_px = state.px + math.cos(v_est.r) * v_est.v
        new_py = state.py + math.sin(v_est.r) * v_est.v
        state = FullState(new_px, new_py, state.vx, state.vy, state.radius,
                          state.pgx, state.pgy, state.v_pref, state.theta+v_est.r)
    else:
        raise ValueError('Type error')

    return state


def build_action_space(v_pref):
    # permissible actions, rotation speed should be smaller than v_pref if the minimum turning radius is 1.0m
    velocities = [i/5*v_pref for i in range(5)]
    rotations = [i/5*math.pi/3 - math.pi/6 for i in range(5)]
    actions = [Action(*x) for x in itertools.product(velocities, rotations)]
    for i in range(10):
        random_velocity = random.random() * v_pref
        random_rotation = random.random() * math.pi/3 - math.pi/6
        actions.append(Action(random_velocity, random_rotation))

    return actions


def run_one_episode(model, phase, env, gamma, epsilon, max_time=1000):
    # observe and take action till the episode is finished
    states = env.reset()
    time_to_goal = 0
    state_sequences = defaultdict(list)
    action_sequences = defaultdict(list)
    done = [False, False]
    while not all(done) and time_to_goal < max_time:
        for agent_idx in range(2):
            if done[agent_idx]:
                action = Action(0, 0)
                state = None
            else:
                state = states[agent_idx]
                v_neighbor_est = filter_velocity(state, state_sequences, agent_idx)
                s_neighbor_est = propagate(ObservableState(*state[9:]), v_neighbor_est)

                max_value = float('-inf')
                best_action = None
                # pick action according to epsilon-greedy
                probability = random.random()
                action_space = build_action_space(state.v_pref)
                if phase == 'train' and probability < epsilon:
                    action = random.choice(action_space)
                    action_sequences[agent_idx].append(action)
                else:
                    for action in action_space:
                        reward = env.compute_reward(agent_idx, action)
                        s_est = propagate(FullState(*state[:9]), action)
                        model_input = torch.Tensor([s_est + s_neighbor_est])
                        value = reward + pow(gamma, state.v_pref) * model(model_input).data.item()
                        if value > max_value:
                            max_value = value
                            best_action = action
                    action = best_action
            state_sequences[agent_idx].append(state)
            action_sequences[agent_idx].append(action)

        # update t and receive new observations
        states, rewards, done = env.step((action_sequences[0][-1], action_sequences[1][-1]))
        time_to_goal += 1

    return time_to_goal, state_sequences


def optimize_batch(model, data_loader, optimizer, lr_scheduler, criterion, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in data_loader:
            lr_scheduler.step()
            inputs, values = data
            inputs = Variable(inputs)
            values = Variable(values)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item()

        logging.info('Loss in epoch {} is {}'.format(epoch, epoch_loss))


def update_memory(memory, state_sequences, gamma):
    # generate state_value pairs and update the memory pool
    for agent_idx in range(2):
        state_sequence = state_sequences[agent_idx]
        last_time_step = sum([state_sequence is not None])
        for step in range(last_time_step):
            state = state_sequence[step]
            value = compute_value(gamma, last_time_step - step + 1, state.v_pref)
            memory.push((state, value))


def train(model, config):
    gamma = config.getfloat('model', 'gamma')
    batch_size = config.getint('train', 'batch_size')
    learning_rate = config.getfloat('train', 'learning_rate')
    step_size = config.getint('train', 'step_size')
    train_episodes = config.getint('train', 'train_episodes')
    test_interval = config.getint('train', 'test_interval')
    test_episodes = config.getint('train', 'test_episodes')
    capacity = config.getint('train', 'capacity')
    epsilon = config.getfloat('train', 'epsilon')
    num_epochs = config.getint('train', 'num_epochs')

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    memory = ReplayMemory(capacity)
    data_loader = DataLoader(memory, batch_size, shuffle=True)
    train_env = ENV(agent_num=2, radius=2, v_pref=2)
    test_env = ENV(agent_num=2, radius=2, v_pref=2)

    for episode in range(train_episodes):
        if episode % test_interval == 0:
            test_time = []
            for i in range(test_episodes):
                time_to_goal, state_sequences = run_one_episode(model, 'test', test_env, gamma, epsilon)
                test_time.append(time_to_goal)
            avg_time = sum(test_time) / len(test_time)
            logging.info('Testing in episode {} has average {} unit time to goal'.format(episode, avg_time))
        time_to_goal, state_sequences = run_one_episode(model, 'train', train_env, gamma, epsilon)
        logging.info('Training in episode {} has {} unit time to goal'.format(episode, time_to_goal))
        update_memory(memory, state_sequences, gamma)
        optimize_batch(model, data_loader, optimizer, lr_scheduler, criterion, num_epochs)

    return model


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    config_file = args.file
    config = configparser.RawConfigParser()
    config.read(config_file)

    # file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    state_dim = config.getint('model', 'state_dim')
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100])
    logging.debug('Trainable parameters: {}'.format([name for name, p in model.named_parameters() if p.requires_grad]))

    # initialize model
    initialized_model = initialize(model, config)
    torch.save(initialized_model.state_dict(), 'data/initialized_model.pth')
    logging.info('Finish initializing model. Model saved')

    # train the model
    # trained_model = train(model, config)
    # torch.save(trained_model.state_dict(), 'data/initialized_model.pth')
    # logging.info('Finish initializing model. Model saved')


if __name__ == '__main__':
    main()


