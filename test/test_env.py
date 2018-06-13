import os
import sys
sys.path.append(os.getcwd())
import configparser
import math
from env import ENV
from utils import Action, JointState


def test_step():
    env_config = configparser.RawConfigParser()
    env_config.read('configs/test_env.config')
    test_env = ENV(env_config)
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
    test_env.reset()
    test_env.step((Action(1, math.pi/2), Action(2, -math.pi/2)))
    test_env.step((Action(4, 0), Action(4, -math.pi)))
    states, rewards, done_signals = test_env.step((Action(1, -math.pi/2), Action(2, math.pi/2)))
    assert rewards == [1, 1]
    assert done_signals == [1, 1]

