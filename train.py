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
import shutil
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


def propagate(state, v_est, kinematic, delta_t=1):
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
        if kinematic:
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


def build_action_space(v_pref, kinematic):
    """
    Action space consists of 25 precomputed actions and 10 randomly sampled actions.

    """
    if kinematic:
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


def run_one_episode(model, phase, env, gamma, epsilon, kinematic, device, seed=None):
    """
    Run two agents simultaneously without communication

    """
    random.seed(seed)
    # observe and take action till the episode is finished
    states = env.reset()
    state_sequences = defaultdict(list)
    state_sequences[0].append(states[0])
    state_sequences[1].append(states[1])
    reward_sequences = defaultdict(list)
    reward_sequences[0].append(0)
    reward_sequences[1].append(0)
    times = [0, 0]
    done = [False, False]
    while not all(done):
        actions = list()
        for agent_idx in range(2):
            state = states[agent_idx]
            if done[agent_idx]:
                # skip an agent which is done already
                actions.append(Action(0, 0))
                continue

            other_v_est = filter_velocity(state, state_sequences, agent_idx)
            other_sn_est = propagate(ObservableState(*state[9:]), other_v_est, kinematic)
            max_value = float('-inf')
            best_action = None
            # pick action according to epsilon-greedy
            probability = random.random()
            action_space = build_action_space(state.v_pref, kinematic)
            if phase == 'train' and probability < epsilon:
                action = random.choice(action_space)
            else:
                for action in action_space:
                    temp_actions = [None] * 2
                    temp_actions[agent_idx] = action
                    reward, _ = env.compute_reward(agent_idx, temp_actions)
                    sn_est = propagate(FullState(*state[:9]), action, kinematic)
                    sn_est = torch.Tensor([sn_est + other_sn_est]).to(device)
                    value = reward + pow(gamma, state.v_pref) * model(sn_est, device).data.item()
                    if value > max_value:
                        max_value = value
                        best_action = action
                action = best_action
            actions.append(action)

        # update t and receive new observations
        states, rewards, done = env.step(actions)
        for agent_idx in range(2):
            state_sequences[agent_idx].append(states[agent_idx])
            reward_sequences[agent_idx].append(rewards[agent_idx])
            times[agent_idx] += 1

    return times, state_sequences, reward_sequences, done


def optimize_batch(model, data_loader, data_size, optimizer, lr_scheduler, criterion, num_epochs, device):
    if lr_scheduler is not None:
        lr_scheduler.step()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in data_loader:
            inputs, values = data
            inputs = Variable(inputs)
            values = Variable(values)

            optimizer.zero_grad()

            outputs = model(inputs, device)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item()
        # logging.info('Loss in epoch {} is {}'.format(epoch, epoch_loss))
        losses.append(epoch_loss / data_size)
    average_epoch_loss = sum(losses) / len(losses)
    return average_epoch_loss


def update_memory(duplicate_model, memory, state_sequences, reward_sequences, gamma, agent_idx, device):
    """
    Estimate state values of finished episode and update the memory pool

    """
    state_sequence0 = state_sequences[agent_idx]
    reward_sequence0 = reward_sequences[agent_idx]
    state_sequence1 = state_sequences[1-agent_idx]
    tg0 = sum([state is not None for state in state_sequence0])
    tg1 = sum([state is not None for state in state_sequence1])
    for step in range(tg0-1):
        state0 = state_sequence0[step]
        next_state0 = state_sequence0[step+1]
        reward0 = reward_sequence0[step]
        # approximate the value with TD prediction based on the next state
        value = reward0 + gamma * duplicate_model(torch.Tensor([next_state0]), device).data.item()

        # penalize non-cooperating behaviors
        state1 = state_sequence1[step]
        if state0 is None:
            te0 = 0
        else:
            te0 = tg0-1-step - np.linalg.norm((state0.px-state0.pgx, state0.py-state0.pgy))/state0.v_pref
        if state1 is None:
            te1 = 0
        else:
            te1 = tg1-1-step - np.linalg.norm((state1.px-state1.pgx, state1.py-state1.pgy))/state1.v_pref
        if te0 < 1 and te1 > 6:
            # TODO: explore different configurations
            value -= 0.1

        state0 = torch.Tensor(state0).to(device)
        value = torch.Tensor([value]).to(device)
        memory.push((state0, value))


def initialize_memory(traj_dir, gamma, capacity, kinematic, device):
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
                                 times, positions, kinematic)
        trajectory2 = Trajectory(gamma, *[float(x) for x in lines[1].split()],
                                 times, positions[:, ::-1, :], kinematic)
        generated_pairs = trajectory1.generate_state_value_pairs(device) + trajectory2.generate_state_value_pairs(device)
        for pair in generated_pairs:
            memory.push(pair)

    logging.info('Total number of state_value pairs: {}'.format(len(memory)))

    return memory


def initialize_model(model, memory, model_config, device):
    num_epochs = model_config.getint('init', 'num_epochs')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')

    data_loader = DataLoader(memory, batch_size, shuffle=True)
    criterion = nn.MSELoss().to(device)
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

            outputs = model(inputs, device)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.item()

        logging.info('Loss in epoch {}: {:.2f}'.format(epoch, epoch_loss))
    return model


def run_k_episodes(num_episodes, episode, model, phase, env, gamma, epsilon, kinematic, duplicate_model, memory, device):
    """
    Run k episodes and measure the average time to goal, access rate and failure rate

    """
    etg = []
    succ = 0
    failure = 0
    # if phase == 'test':
    #     seed = 0
    # else:
    #     seed = time.time()
    for _ in range(num_episodes):
        times, state_sequences, reward_sequences, end_signals = run_one_episode(model, phase, env, gamma,
                                                                                epsilon, kinematic, device)
        # success is defined on the group's success
        if end_signals[0] == 1 and end_signals[1] == 1:
            succ += 1
            etg.append(sum(times) / len(times) - 4)
        if end_signals[0] == 2 and end_signals[1] == 2:
            failure += 1
        if duplicate_model is not None and memory is not None:
            update_memory(duplicate_model, memory, state_sequences, reward_sequences, gamma, 0, device)
            update_memory(duplicate_model, memory, state_sequences, reward_sequences, gamma, 1, device)

    if len(etg) == 0:
        average_time = 0
    else:
        average_time = sum(etg) / len(etg)
    logging.info('{} in episode {} has success rate: {:.2f}, failure rate: {:.2f}, average extra time to goal: {:.0f}'.
                 format(phase, episode, succ / num_episodes, failure / num_episodes, average_time))

    return etg, succ, failure


def train(model, memory, model_config, env_config, device, weight_file):
    gamma = model_config.getfloat('model', 'gamma')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')
    train_episodes = model_config.getint('train', 'train_episodes')
    sample_episodes = model_config.getint('train', 'sample_episodes')
    test_interval = model_config.getint('train', 'test_interval')
    test_episodes = model_config.getint('train', 'test_episodes')
    epsilon_start = model_config.getfloat('train', 'epsilon_start')
    epsilon_end = model_config.getfloat('train', 'epsilon_end')
    epsilon_decay = model_config.getfloat('train', 'epsilon_decay')
    num_epochs = model_config.getint('train', 'num_epochs')
    kinematic = env_config.getboolean('agent', 'kinematic')
    checkpoint_interval = model_config.getint('train', 'checkpoint_interval')

    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(memory, batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    train_env = ENV(config=env_config, phase='train')
    test_env = ENV(config=env_config, phase='test')
    duplicate_model = copy.deepcopy(model)

    episode = 0
    while episode < train_episodes:
        # epsilon-greedy
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end

        # test
        if episode % test_interval == 0:
            run_k_episodes(test_episodes, episode, model, 'test', test_env, gamma, epsilon,
                           kinematic, None, None, device)
            # update duplicate model
            duplicate_model = copy.deepcopy(model)

        # sample k episodes into memory and optimize over the generated memory
        run_k_episodes(sample_episodes, episode, model, 'train', train_env, gamma, epsilon,
                       kinematic, duplicate_model, memory, device)
        optimize_batch(model, data_loader, len(memory), optimizer, None, criterion, num_epochs, device)
        episode += 1

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), weight_file)

    return model


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/model.config')
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args()
    config_file = args.config
    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    # configure paths
    output_dir = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = os.path.join('data', output_dir)
    if os.path.exists(output_dir):
        # raise FileExistsError('Output folder already exists')
        print('Output folder already exists')
    else:
        os.mkdir(output_dir)
    log_file = os.path.join(output_dir, 'output.log')
    shutil.copy(args.config, output_dir)
    initialized_weights = os.path.join(output_dir, 'initialized_model.pth')
    trained_weights = os.path.join(output_dir, 'trained_model.pth')

    # configure logging
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # configure device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: {}'.format(device))

    # configure model
    state_dim = model_config.getint('model', 'state_dim')
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100]).to(device)
    logging.debug('Trainable parameters: {}'.format([name for name, p in model.named_parameters() if p.requires_grad]))

    # load simulated data from ORCA
    traj_dir = model_config.get('init', 'traj_dir')
    gamma = model_config.getfloat('model', 'gamma')
    kinematic = env_config.getboolean('agent', 'kinematic')
    capacity = model_config.getint('train', 'capacity')
    memory = initialize_memory(traj_dir, gamma, capacity, kinematic, device)

    # initialize model
    if os.path.exists(initialized_weights):
        model.load_state_dict(torch.load(initialized_weights))
        logging.info('Load initialized model weights')
    else:
        initialize_model(model, memory, model_config, device)
        torch.save(model.state_dict(), initialized_weights)
        logging.info('Finish initializing model. Model saved')

    # train the model
    train(model, memory, model_config, env_config, device, trained_weights)
    torch.save(model.state_dict(), trained_weights)
    logging.info('Finish initializing model. Model saved')


if __name__ == '__main__':
    main()


