import os
import sys
sys.path.append(os.getcwd())
from model import ValueNetwork
from env import JointState
import torch
import numpy as np


def test_rotate():
    vn = ValueNetwork(14, [150, 150, 100], kinematic=False)
    state = JointState(2, 2, 0, 1, 0.3, 2, 4, 1, 0, 4, 2, 2, 0, 0.3)
    state = torch.Tensor(state).expand(1, 14)
    rotated_state = vn.rotate(state, torch.device('cpu')).squeeze().numpy()
    assert np.allclose(rotated_state, [2, 1, 1, 0, 0.3, 0, 0, -2, 0, -2, 0.3, 0.6, 1, 0, 2], atol=1e-06)

    vn = ValueNetwork(14, [150, 150, 100], kinematic=True)
    state = JointState(2, 2, 0, 1, 0.3, 2, 4, 1, 0, 4, 2, 2, 0, 0.3)
    state = torch.Tensor(state).expand(1, 14)
    rotated_state = vn.rotate(state, torch.device('cpu')).squeeze().numpy()
    assert np.allclose(rotated_state, [2, 1, 1, 0, 0.3, -np.pi/2, 0, -2, 0, -2, 0.3, 0.6, 0, -1, 2], atol=1e-06)


test_rotate()