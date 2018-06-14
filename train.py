import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy
import sys
import logging
import random
import itertools
import argparse
import configparser
import math
import os
import numpy as np
import re
import time
from collections import defaultdict
from model import ValueNetwork
from env import ENV
from utils import *


def filter_velocity(joint_state, state_sequences, agent_idx):
    """
    Compute the other agent's average velocity in last two time steps

    """
    # TODO: filter speed
    # if agent_idx not in state_sequences:
    #     prev_v = Velocity(0, 0)
    # else:
    #     prev_v = Velocity(state_sequences[agent_idx][-1].vx1, state_sequences[agent_idx][-1].vy1)
    # current_v = Velocity(joint_state.vx1, joint_state.vy1)
    # filtered_v = Velocity((prev_v.x+current_v.x)/2, (prev_v.y+current_v.y)/2)
    filtered_v = Velocity(joint_state.vx1, joint_state.vy1)

    return filtered_v


def propagate(state, v_est, kinematic_constrained, delta_t=1):
    """
    Compute approximate next state with estimated velocity/action

    """
    if isinstance(state, ObservableState) and isinstance(v_est, Velocity):
        # propagate state of the other agent
        new_px = state.px + v_est.x * delta_t
        new_py = state.py + v_est.y * delta_t
        state = ObservableState(new_px, new_py, v_est.x, v_est.y, state.radius)
    elif isinstance(state, FullState) and isinstance(v_est, Action):
        # propagate state of current agent
        # perform action without rotation
        if kinematic_constrained:
            # TODO: impose kinematic constraint and theta
            pass
        else:
            new_px = state.px + math.cos(v_est.r) * v_est.v * delta_t
            new_py = state.py + math.sin(v_est.r) * v_est.v * delta_t
            state = FullState(new_px, new_py, state.vx, state.vy, state.radius,
                              state.pgx, state.pgy, state.v_pref, state.theta)
    else:
        raise ValueError('Type error')

    return state


def build_action_space(v_pref, kinematic_constrained):
    """
    Action space consists of 25 precomputed actions and 10 randomly sampled actions.

    """
    if kinematic_constrained:
        velocities = [i/4*v_pref for i in range(5)]
        rotations = [i/4*math.pi/3 - math.pi/6 for i in range(5)]
        actions = [Action(*x) for x in itertools.product(velocities, rotations)]
        for i in range(10):
            random_velocity = random.random() * v_pref
            random_rotation = random.random() * math.pi/3 - math.pi/6
            actions.append(Action(random_velocity, random_rotation))
    else:
        velocities = [(i+1)/5*v_pref for i in range(5)]
        rotations = [i/4*2*math.pi for i in range(5)]
        actions = [Action(*x) for x in itertools.product(velocities, rotations)]
        for i in range(25):
            random_velocity = random.random() * v_pref
            random_rotation = random.random() * 2 * math.pi
            actions.append(Action(random_velocity, random_rotation))
        actions.append(Action(0, 0))

    return actions


def run_one_episode(model, phase, env, gamma, epsilon, kinematic_constrained, seed=None):
    """
    Run two agents simultaneously without communication

    """
    random.seed(time.time())
    # observe and take action till the episode is finished
    states = env.reset()
    state_sequences = defaultdict(list)
    state_sequences[0].append(states[0])
    state_sequences[1].append(states[1])
    times = [0, 0]
    end_signals = [0, 0]
    done = [False, False]
    while not all(done):
        actions = list()
        for agent_idx in range(2):
            state = states[agent_idx]
            if done[agent_idx]:
                action = Action(0, 0)
            else:
                v1_est = filter_velocity(state, state_sequences, agent_idx)
                s1n_est = propagate(ObservableState(*state[9:]), v1_est, kinematic_constrained)

                max_value = float('-inf')
                best_action = None
                # pick action according to epsilon-greedy
                probability = random.random()
                action_space = build_action_space(state.v_pref, kinematic_constrained)
                if phase == 'train' and probability < epsilon:
                    action = random.choice(action_space)
                else:
                    for action in action_space:
                        reward, _ = env.compute_reward(agent_idx, [action, None])
                        s0n_est = propagate(FullState(*state[:9]), action, kinematic_constrained)
                        s0n_est = torch.Tensor([s0n_est + s1n_est])
                        value = reward + pow(gamma, state.v_pref) * model(s0n_est).data.item()
                        if value > max_value:
                            max_value = value
                            best_action = action
                    action = best_action
            actions.append(action)

        # update t and receive new observations
        last_done = done
        states, rewards, done = env.step(actions)
        for agent_idx in range(2):
            if last_done[agent_idx]:
                state_sequences[agent_idx].append(None)
            else:
                if done[agent_idx]:
                    end_signals[agent_idx] = done[agent_idx]
                state_sequences[agent_idx].append(states[agent_idx])
                times[agent_idx] += 1

    return times, state_sequences, end_signals


def optimize_batch(model, data_loader, optimizer, lr_scheduler, criterion, num_epochs):
    lr_scheduler.step()
    for epoch in range(num_epochs):
        epoch_loss = 0
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
        logging.info('Loss in epoch {} is {}'.format(epoch, epoch_loss))


def update_memory(duplicate_model, memory, state_sequences, agent_idx):
    """
    Estimate state values of finished episode and update the memory pool

    """
    state_sequence0 = state_sequences[agent_idx]
    state_sequence1 = state_sequences[1-agent_idx]
    tg0 = sum([state is not None for state in state_sequence0])
    tg1 = sum([state is not None for state in state_sequence1])
    for step in range(tg0):
        state0 = state_sequence0[step]
        value = duplicate_model(torch.Tensor(state0)).data.numpy()

        # penalize aggressive behaviors
        state1 = state_sequence1[step]
        te0 = tg0-1-step - np.linalg.norm((state0.px-state0.pgx, state0.py-state1.pgy))/state0.v_pref
        te1 = tg1-1-step - np.linalg.norm((state1.px-state0.pgx, state1.py-state1.pgy))/state1.v_pref
        if te0 < 1 and te1 > 2**4:
            value -= 0.1

        memory.push((state0, value))


def initialize_memory(traj_dir, gamma, capacity, kinematic_constrained):
    memory = ReplayMemory(capacity=capacity)
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
        generated_pairs = trajectory1.generate_state_value_pairs() + trajectory2.generate_state_value_pairs()
        for pair in generated_pairs:
            memory.push(pair)

    logging.info('Total number of state_value pairs: {}'.format(len(memory)))

    return memory


def initialize_model(model, memory, model_config):
    num_epochs = model_config.getint('init', 'num_epochs')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')

    data_loader = DataLoader(memory, batch_size, shuffle=True)
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


def train(model, memory, model_config, env_config):
    gamma = model_config.getfloat('model', 'gamma')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')
    train_episodes = model_config.getint('train', 'train_episodes')
    test_interval = model_config.getint('train', 'test_interval')
    test_episodes = model_config.getint('train', 'test_episodes')
    epsilon_start = model_config.getfloat('train', 'epsilon_start')
    epsilon_end = model_config.getfloat('train', 'epsilon_end')
    epsilon_decay = model_config.getfloat('train', 'epsilon_decay')
    num_epochs = model_config.getint('train', 'num_epochs')
    kinematic_constrained = env_config.getboolean('agent', 'kinematic_constrained')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    data_loader = DataLoader(memory, batch_size, shuffle=True)
    train_env = ENV(config=env_config)
    test_env = ENV(config=env_config)
    duplicate_model = copy.deepcopy(model)

    episode = 0
    while episode < train_episodes:
        if episode % test_interval == 0:
            # test_time = []
            # for i in range(test_episodes):
            #     time_to_goal, state_sequences, success = run_one_episode(model, 'test', test_env, gamma,
            #                                                              None, kinematic_constrained)
            #     test_time.append(time_to_goal)
            # avg_time = sum(test_time) / len(test_time)
            # logging.info('Testing in episode {} has average {} unit time to goal'.format(episode, avg_time))

            # update duplicate model
            duplicate_model = copy.deepcopy(model)

        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        times, state_sequences, end_signals = run_one_episode(model, 'train', train_env, gamma,
                                                              epsilon, kinematic_constrained, seed=None)

        # update the memory if the episode runs to the end (reach the goal or collide)
        for agent_idx in range(2):
            if end_signals[agent_idx] in [1, 2]:
                logging.info('Agent {} in episode {} reaches goal in {}'.format(agent_idx, episode, times[agent_idx]))
                update_memory(duplicate_model, memory, state_sequences, agent_idx)
                optimize_batch(model, data_loader, optimizer, lr_scheduler, criterion, num_epochs)
                episode += 1

    return model


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--file', type=str, default='configs/model.config')
    args = parser.parse_args()
    config_file = args.file
    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    file_handler = logging.FileHandler('output.log', mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    state_dim = model_config.getint('model', 'state_dim')
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100])
    logging.debug('Trainable parameters: {}'.format([name for name, p in model.named_parameters() if p.requires_grad]))

    # load simulated data from ORCA
    traj_dir = model_config.get('init', 'traj_dir')
    gamma = model_config.getfloat('model', 'gamma')
    kinematic_constrained = env_config.getboolean('agent', 'kinematic_constrained')
    capacity = model_config.getint('train', 'capacity')
    memory = initialize_memory(traj_dir, gamma, capacity, kinematic_constrained)

    # initialize model
    initialized_model = initialize_model(model, memory, model_config)
    torch.save(initialized_model.state_dict(), 'data/initialized_model.pth')
    logging.info('Finish initializing model. Model saved')
    # model.load_state_dict(torch.load('data/initialized_model.pth'))

    # train the model
    # trained_model = train(model, memory, model_config, env_config)
    # torch.save(trained_model.state_dict(), 'data/trained_model.pth')
    # logging.info('Finish initializing model. Model saved')


if __name__ == '__main__':
    main()


