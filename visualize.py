import argparse
import configparser
import torch
import os
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from model import ValueNetwork
from env import ENV
from train import run_one_episode


def visualize(model_config, env_config, weight_path, case, save):
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
    kinematic = env_config.getboolean('agent', 'kinematic')
    radius = env_config.getfloat('agent', 'radius')

    device = torch.device('cpu')
    test_env = ENV(config=env_config, phase='test')
    test_env.reset(case)
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100], kinematic=kinematic)
    model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
    _, state_sequences, _, _ = run_one_episode(model, 'test', test_env, gamma, None, kinematic, device)

    positions = list()
    colors = list()
    counter = list()
    line_positions = list()
    for i in range(len(state_sequences[0])):
        counter.append(i)
        if state_sequences[0][i] is None:
            p0 = positions[-1][0]
            c0 = 'tab:red'
            h0 = 0
        else:
            p0 = (state_sequences[0][i].px, state_sequences[0][i].py)
            c0 = 'tab:blue'
            h0 = state_sequences[0][i].theta
        xdata0 = [p0[0], p0[0]+radius*np.cos(h0)]
        ydata0 = [p0[1], p0[1]+radius*np.sin(h0)]
        if state_sequences[1][i] is None:
            p1 = positions[-1][1]
            c1 = 'tab:red'
            h1 = 0
        else:
            p1 = (state_sequences[1][i].px, state_sequences[1][i].py)
            c1 = 'tab:gray'
            h1 = state_sequences[1][i].theta
        xdata1 = [p1[0], p1[0]+radius*np.cos(h1)]
        ydata1 = [p1[1], p1[1]+radius*np.sin(h1)]
        if i == len(state_sequences[0])-1:
            c0 = c1 = 'tab:red'
        positions.append([p0, p1])
        colors.append([c0, c1])
        line_positions.append([[xdata0, ydata0], [xdata1, ydata1]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.add_artist(plt.Circle((0, 0), crossing_radius, fill=False, edgecolor='g', lw=1))
    ax.add_artist(plt.Rectangle((bxmin, bymin), bxmax-bxmin, bymax-bymin, fill=False, linestyle='dashed', lw=1))
    agent0 = plt.Circle(positions[0][0], radius, fill=True, color='b')
    agent1 = plt.Circle(positions[0][1], radius, fill=True, color='c')
    line0 = plt.Line2D(line_positions[0][0][0], line_positions[0][0][1], color='tab:red')
    line1 = plt.Line2D(line_positions[0][1][0], line_positions[0][1][1], color='tab:red')
    text = plt.text(0, 8, 'Step: {}'.format(counter[0]), fontsize=12)
    ax.add_artist(agent0)
    ax.add_artist(agent1)
    ax.add_artist(line0)
    ax.add_artist(line1)
    ax.add_artist(text)

    def update(frame_num):
        agent0.center = positions[frame_num][0]
        agent1.center = positions[frame_num][1]
        agent0.set_color(colors[frame_num][0])
        agent1.set_color(colors[frame_num][1])
        line0.set_xdata(line_positions[frame_num][0][0])
        line0.set_ydata(line_positions[frame_num][0][1])
        line1.set_xdata(line_positions[frame_num][1][0])
        line1.set_ydata(line_positions[frame_num][1][1])
        text.set_text('Step: {}'.format(counter[frame_num]))

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=800)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        output_file = 'data/output.mp4'
        anim.save(output_file, writer=writer)

    plt.show()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--init', default=False, action='store_true')
    parser.add_argument('--case', default=0, type=int)
    parser.add_argument('--save', default=False, action='store_true')
    args = parser.parse_args()
    config_file = os.path.join(args.output_dir, 'model.config')
    if args.init:
        weight_file = os.path.join(args.output_dir, 'initialized_model.pth')
    else:
        weight_file = os.path.join(args.output_dir, 'trained_model.pth')

    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    visualize(model_config, env_config, weight_file, args.case, args.save)


if __name__ == '__main__':
    main()