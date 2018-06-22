import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from utils import Trajectory


def test_trajectory():
    gamma = 0.9
    goal_x = 2
    goal_y = 0
    radius = 0.3
    v_pref = 1.0
    times = [0, 1, 2, 3, 4]
    positions = [[(0, 0), (2, 0)], [(0, 1), (2, -1)], [(1, 1), (1, -1)], [(2, 1), (0, -1)], [(2, 0), (0, 0)]]
    positions = np.array(positions)

    traj = Trajectory(gamma, goal_x, goal_y, radius, v_pref, times, positions, False)
    state_values_pairs = traj.generate_state_value_pairs(torch.device('cpu'))

    assert len(state_values_pairs) == len(times) - 1
    assert np.allclose(state_values_pairs[0][0].numpy(), (0, 1, 0, 1, 0.3, 2, 0, 1, 0, 2, -1, 0, -1, 0.3))
    assert np.isclose(state_values_pairs[0][1], pow(gamma, 3))
    print(state_values_pairs[-1][0].numpy())
    assert np.allclose(state_values_pairs[-1][0].numpy(), (2, 0, 0, -1, 0.3, 2, 0, 1, 0, 0, 0, 0, 1, 0.3))
    assert np.isclose(state_values_pairs[-1][1], 1)

    traj = Trajectory(gamma, goal_x, goal_y, radius, v_pref, times, positions, True)
    state_values_pairs = traj.generate_state_value_pairs(torch.device('cpu'))

    assert len(state_values_pairs) == len(times) - 1
    assert np.allclose(state_values_pairs[0][0].numpy(), (0, 1, 0, 1, 0.3, 2, 0, 1, np.pi/2, 2, -1, 0, -1, 0.3))
    assert np.isclose(state_values_pairs[0][1], pow(gamma, 3))
    print(state_values_pairs[-1][0].numpy())
    assert np.allclose(state_values_pairs[-1][0].numpy(), (2, 0, 0, -1, 0.3, 2, 0, 1, -np.pi/2, 0, 0, 0, 1, 0.3))
    assert np.isclose(state_values_pairs[-1][1], 1)