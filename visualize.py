import argparse
import configparser
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import ValueNetwork
from env import ENV
from train import run_one_episode


def visualize(model_config, env_config, weight_path):
    state_dim = model_config.getint('model', 'state_dim')
    gamma = model_config.getfloat('model', 'gamma')
    bxmin = env_config.getfloat('sim', 'xmin')
    bxmax = env_config.getfloat('sim', 'xmax')
    bymin = env_config.getfloat('sim', 'ymin')
    bymax = env_config.getfloat('sim', 'ymax')
    xmin = env_config.getfloat('visualization', 'xmin')
    xmax = env_config.getfloat('visualization', 'xmax')
    ymin = env_config.getfloat('visualization', 'ymin')
    ymax = env_config.getfloat('visualization', 'ymax')
    crossing_radius = env_config.getfloat('sim', 'crossing_radius')
    kinematic_constrained = env_config.getboolean('agent', 'kinematic_constrained')
    radius = env_config.getfloat('agent', 'radius')

    test_env = ENV(config=env_config)
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100])
    model.load_state_dict(torch.load(weight_path))
    _, state_sequences, _ = run_one_episode(model, 'test', test_env, gamma, None, kinematic_constrained)

    positions = list()
    colors = list()
    counter = list()
    for i in range(len(state_sequences[0])):
        counter.append(i)
        if state_sequences[0][i] is None:
            p0 = positions[-1][0]
            c0 = 'tab:red'
        else:
            p0 = (state_sequences[0][i].px, state_sequences[0][i].py)
            c0 = 'tab:blue'
        if state_sequences[1][i] is None:
            p1 = positions[-1][1]
            c1 = 'tab:red'
        else:
            p1 = (state_sequences[1][i].px, state_sequences[1][i].py)
            c1 = 'tab:gray'
        if i == len(state_sequences[0])-1:
            c0 = c1 = 'tab:red'
        positions.append([p0, p1])
        colors.append([c0, c1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.add_artist(plt.Circle((0, 0), crossing_radius, fill=False, edgecolor='g', lw=1))
    ax.add_artist(plt.Rectangle((bxmin, bymin), bxmax-bxmin, bymax-bymin, fill=False, linestyle='dashed', lw=1))
    agent0 = plt.Circle(positions[0][0], radius, fill=True, color='b')
    agent1 = plt.Circle(positions[0][1], radius, fill=True, color='c')
    text = plt.text(0, 8, 'Step: {}'.format(counter[0]), fontsize=12)
    ax.add_artist(agent0)
    ax.add_artist(agent1)
    ax.add_artist(text)

    def update(frame_num):
        agent0.center = positions[frame_num][0]
        agent1.center = positions[frame_num][1]
        agent0.set_color(colors[frame_num][0])
        agent1.set_color(colors[frame_num][1])
        text.set_text('Step: {}'.format(counter[frame_num]))

    animation = FuncAnimation(fig, update, frames=len(positions), interval=800)
    plt.show()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    config_file = os.path.join(args.output_dir, 'model.config')
    weight_file = os.path.join(args.output_dir, 'initialized_model.pth')

    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    visualize(model_config, env_config, weight_file)


if __name__ == '__main__':
    main()