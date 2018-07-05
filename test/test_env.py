import os
import sys
sys.path.append(os.getcwd())
import configparser
import numpy as np
from env import ENV
from utils import Action, JointState


def test_step_without_kinematic():
    env_config = configparser.RawConfigParser()
    env_config.read('configs/test_env.config')
    env_config.set('agent', 'kinematic', 'false')
    test_env = ENV(env_config, phase='test')
    test_env.reset()

    # test state computation
    states, rewards, done_signals = test_env.step((Action(1, 0), Action(-1, 0)))
    assert states[0] == JointState(-1, 0, 1, 0, 0.3, 2, 0, 1.0, 0, 1, 0, -1, 0, 0.3)
    assert states[1] == JointState(1, 0, -1, 0, 0.3, -2, 0, 1.0, 0, -1, 0, 1, 0, 0.3)
    assert rewards == [0, 0]
    assert done_signals == [False, False]

    # test one-step lookahead
    reward, end_time = test_env.compute_reward(0, [Action(1.5, 0), None])
    assert reward == -0.25
    assert end_time == 1

    reward, end_time = test_env.compute_reward(0, [Action(1.5, 0), Action(-1.5, 0)])
    assert reward == -0.25
    assert end_time == 0.5

    # test collision detection
    states, rewards, done_signals = test_env.step((Action(1, 0), Action(-1, 0)))
    assert states[0] == JointState(0, 0, 1, 0, 0.3, 2, 0, 1.0, 0, 0, 0, -1, 0, 0.3)
    assert states[1] == JointState(0, 0, -1, 0, 0.3, -2, 0, 1.0, 0, 0, 0, 1, 0, 0.3)
    assert rewards == [-0.25, -0.25]
    assert done_signals == [2, 2]

    # test reaching goal
    test_env = ENV(env_config, phase='test')
    test_env.reset()
    test_env.step((Action(1, np.pi/2), Action(2, -np.pi/2)))
    test_env.step((Action(4, 0), Action(4, -np.pi)))
    states, rewards, done_signals = test_env.step((Action(1, -np.pi/2), Action(2, np.pi/2)))
    assert rewards == [1, 1]
    assert done_signals == [1, 1]


def test_step_with_kinematic():
    env_config = configparser.RawConfigParser()
    env_config.read('configs/test_env.config')
    env_config.set('agent', 'kinematic', 'true')
    test_env = ENV(env_config, phase='test')
    test_env.reset()

    # test state computation
    states, rewards, done_signals = test_env.step((Action(1, 0), Action(1, 0)))
    assert np.allclose(states[0], JointState(-1, 0, 1, 0, 0.3, 2, 0, 1.0, 0, 1, 0, -1, 0, 0.3))
    assert np.allclose(states[1], JointState(1, 0, -1, 0, 0.3, -2, 0, 1.0, np.pi, -1, 0, 1, 0, 0.3))
    assert rewards == [0, 0]
    assert done_signals == [False, False]

    # test one-step lookahead
    reward, end_time = test_env.compute_reward(0, [Action(1.5, 0), None])
    assert reward == -0.25
    assert end_time == 1

    reward, end_time = test_env.compute_reward(0, [Action(1.5, 0), Action(1.5, 0)])
    assert reward == -0.25
    assert end_time == 0.5

    # test collision detection
    states, rewards, done_signals = test_env.step((Action(1, 0), Action(1, 0)))
    assert np.allclose(states[0], JointState(0, 0, 1, 0, 0.3, 2, 0, 1.0, 0, 0, 0, -1, 0, 0.3))
    assert np.allclose(states[1], JointState(0, 0, -1, 0, 0.3, -2, 0, 1.0, np.pi, 0, 0, 1, 0, 0.3))
    assert rewards == [-0.25, -0.25]
    assert done_signals == [2, 2]

    # test reaching goal
    test_env = ENV(env_config, phase='test')
    test_env.reset()
    test_env.step((Action(1, np.pi/2), Action(2, np.pi/2)))
    test_env.step((Action(4, -np.pi/2), Action(4, -np.pi/2)))
    states, rewards, done_signals = test_env.step((Action(1, -np.pi/2), Action(2, -np.pi/2)))
    assert rewards == [1, 1]
    assert done_signals == [1, 1]

