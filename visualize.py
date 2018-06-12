import argparse
import configparser
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import ValueNetwork
from env import ENV
from train import run_one_episode


def visualize(model_config, env_config, weight_path):
    state_dim = model_config.getint('model', 'state_dim')
    gamma = model_config.getfloat('model', 'gamma')
    epsilon = model_config.getfloat('train', 'epsilon')
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

    test_env = ENV(config=env_config)
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100])
    model.load_state_dict(torch.load(weight_path))
    _, state_sequences = run_one_episode(model, 'test', test_env, gamma, epsilon, kinematic_constrained)

    positions = list()
    colors = list()
    for i in range(len(state_sequences[0])):
        if state_sequences[0][i] is None:
            p0 = positions[-1][0]
            c0 = 'r'
        else:
            p0 = (state_sequences[0][i].px, state_sequences[0][i].py)
            c0 = 'b'
        if state_sequences[1][i] is None:
            p1 = positions[-1][1]
            c1 = 'r'
        else:
            p1 = (state_sequences[1][i].px, state_sequences[1][i].py)
            c1 = 'b'
        if i == len(state_sequences[0])-1:
            c0 = c1 = 'r'
        positions.append([p0, p1])
        colors.append([c0, c1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    scat = ax.scatter([x[0] for x in positions[0]], [x[1] for x in positions[0]], s=500)
    ax.add_artist(plt.Circle((0, 0), crossing_radius, fill=False, edgecolor='g', lw=1))
    ax.add_artist(plt.Rectangle((bxmin, bymin), bxmax-bxmin, bymax-bymin, fill=False, linestyle='dashed', lw=1))

    def update(frame_num):
        scat.set_offsets(positions[frame_num])
        scat.set_color(colors[frame_num])

    animation = FuncAnimation(fig, update, frames=len(positions), interval=800)
    plt.show()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--file', type=str, default='configs/model.config')
    parser.add_argument('--weight_path', type=str, default='data/initialized_model.pth')
    args = parser.parse_args()
    config_file = args.file
    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    visualize(model_config, env_config, args.weight_path)


if __name__ == '__main__':
    main()