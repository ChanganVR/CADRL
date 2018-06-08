import argparse
import configparser
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import ValueNetwork
from env import ENV
from train import run_one_episode


def visualize(config, weight_path):
    state_dim = config.getint('model', 'state_dim')
    gamma = config.getfloat('model', 'gamma')
    epsilon = config.getfloat('train', 'epsilon')

    test_env = ENV(agent_num=2, radius=2, v_pref=2)
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100])
    model.load_state_dict(torch.load(weight_path))
    _, state_sequences = run_one_episode(model, 'test', test_env, gamma, epsilon)

    steps = list()
    for i in range(len(state_sequences[0])):
        if state_sequences[0][i] is None:
            p0 = steps[-1][0]
        else:
            p0 = (state_sequences[0][i].px, state_sequences[0][i].py)
        if state_sequences[1][i] is None:
            p1 = steps[-1][1]
        else:
            p1 = (state_sequences[1][i].px, state_sequences[1][i].py)
        steps.append([p0, p1])
    print(steps)

    xmin = min([x[0] for step in steps for x in step])
    xmax = max([x[0] for step in steps for x in step])
    ymin = min([x[1] for step in steps for x in step])
    ymax = max([x[1] for step in steps for x in step])
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    scat = ax.scatter([x[0] for x in steps[0]], [x[1] for x in steps[0]])

    def update(frame_num):
        scat.set_offsets(steps[frame_num])

    animation = FuncAnimation(fig, update, frames=len(steps), interval=1)
    plt.show()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('file', type=str)
    parser.add_argument('weight_path', type=str)
    args = parser.parse_args()
    config_file = args.file
    config = configparser.RawConfigParser()
    config.read(config_file)

    visualize(config, args.weight_path)


if __name__ == '__main__':
    main()