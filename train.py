import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
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
        # TODO: how to propagate
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
                              state.pgx, state.pgy, state.v_pref, state.theta + v_est.r)
    else:
        raise ValueError('Type error')

    return state


def build_action_space(v_pref, kinematic_constrained):
    """
    Action space consists of 25 precomputed actions and 10 randomly sampled actions.

    """
    random.seed(0)
    if kinematic_constrained:
        velocities = [i/5*v_pref for i in range(5)]
        rotations = [i/5*math.pi/3 - math.pi/6 for i in range(5)]
        actions = [Action(*x) for x in itertools.product(velocities, rotations)]
        for i in range(10):
            random_velocity = random.random() * v_pref
            random_rotation = random.random() * math.pi/3 - math.pi/6
            actions.append(Action(random_velocity, random_rotation))
    else:
        velocities = [i/5*v_pref for i in range(5)]
        rotations = [i/5*2*math.pi for i in range(5)]
        actions = [Action(*x) for x in itertools.product(velocities, rotations)]
        for i in range(25):
            random_velocity = random.random() * v_pref
            random_rotation = random.random() * 2 * math.pi
            actions.append(Action(random_velocity, random_rotation))

    return actions


def run_one_episode(model, phase, env, gamma, epsilon, kinematic_constrained):
    # observe and take action till the episode is finished
    states = env.reset()
    time_to_goal = 0
    state_sequences = defaultdict(list)
    action_sequences = defaultdict(list)
    done = [False, False]
    while not all(done):
        for agent_idx in range(2):
            if done[agent_idx]:
                action = Action(0, 0)
                state = None
            else:
                state = states[agent_idx]
                assert state is not None
                v_neighbor_est = filter_velocity(state, state_sequences, agent_idx)
                s_neighbor_est = propagate(ObservableState(*state[9:]), v_neighbor_est, kinematic_constrained)

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
                        s_est = propagate(FullState(*state[:9]), action, kinematic_constrained)
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

    success = done[0] == 1 and done[1] == 1
    if not success:
        time_to_goal = 100

    return time_to_goal, state_sequences, success


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


def update_memory(memory, state_sequences, gamma):
    # generate state_value pairs and update the memory pool
    for agent_idx in range(2):
        state_sequence = state_sequences[agent_idx]
        last_time_step = sum([state_sequence is not None])
        for step in range(last_time_step):
            state = state_sequence[step]
            value = compute_value(gamma, last_time_step - step + 1, state.v_pref)
            memory.push((state, value))


def train(model, model_config, env_config):
    gamma = model_config.getfloat('model', 'gamma')
    batch_size = model_config.getint('train', 'batch_size')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    step_size = model_config.getint('train', 'step_size')
    train_episodes = model_config.getint('train', 'train_episodes')
    test_interval = model_config.getint('train', 'test_interval')
    test_episodes = model_config.getint('train', 'test_episodes')
    capacity = model_config.getint('train', 'capacity')
    epsilon_start = model_config.getfloat('train', 'epsilon_start')
    epsilon_end = model_config.getfloat('train', 'epsilon_end')
    epsilon_decay = model_config.getfloat('train', 'epsilon_decay')
    num_epochs = model_config.getint('train', 'num_epochs')
    kinematic_constrained = env_config.getboolean('agent', 'kinematic_constrained')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    memory = ReplayMemory(capacity)
    data_loader = DataLoader(memory, batch_size, shuffle=True)
    train_env = ENV(config=env_config)
    test_env = ENV(config=env_config)

    episode = 0
    while episode < train_episodes:
        # if episode % test_interval == 0:
        #     test_time = []
        #     for i in range(test_episodes):
        #         time_to_goal, state_sequences, success = run_one_episode(model, 'test', test_env, gamma,
        #                                                                  epsilon, kinematic_constrained)
        #         test_time.append(time_to_goal)
        #     avg_time = sum(test_time) / len(test_time)
        #     logging.info('Testing in episode {} has average {} unit time to goal'.format(episode, avg_time))

        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        time_to_goal, state_sequences, success = run_one_episode(model, 'train', train_env, gamma,
                                                                 epsilon, kinematic_constrained)
        if success:
            logging.info('Training in episode {} has {} unit time to goal'.format(episode, time_to_goal))
            update_memory(memory, state_sequences, gamma)
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

    # initialize model
    # initialized_model = initialize(model, model_config, env_config)
    # torch.save(initialized_model.state_dict(), 'data/initialized_model.pth')
    # logging.info('Finish initializing model. Model saved')
    model.load_state_dict(torch.load('data/initialized_model.pth'))

    # train the model
    trained_model = train(model, model_config, env_config)
    torch.save(trained_model.state_dict(), 'data/trained_model.pth')
    logging.info('Finish initializing model. Model saved')


if __name__ == '__main__':
    main()


