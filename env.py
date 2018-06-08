from collections import namedtuple
import math


FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])
JointState = namedtuple('JointState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta',
                                       'px1', 'py1', 'vx1', 'vy1', 'radius1'])
Velocity = namedtuple('Velocity', ['x', 'y'])
# v is velocity, TODO:r is rotation speed or angle?
Action = namedtuple('Action', ['v', 'r'])


class Agent(object):
    def __init__(self, px, py, gx, gy, radius, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

    def update_state(self, action):
        self.px, self.py = self.compute_position(action, 1)
        self.vx = math.cos(action.v)
        self.vy = math.sin(action.v)
        self.theta += action.r

    def get_full_state(self):
        return self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta

    def get_visible_state(self):
        return self.px, self.py, self.vx, self.vy, self.radius

    def compute_position(self, action, time, linear=True):
        if linear:
            # assume the agent travels in linear speed
            x = self.px + time * self.vx
            y = self.py + time * self.vy
        else:
            if action.r == 0:
                x = self.px + time * action.v * math.cos(self.theta)
                y = self.py + time * action.v * math.sin(self.theta)
            else:
                x = self.px + action.v / pow(action.r, 2) * (time * math.sin(self.theta + action.r * time) +
                                                             math.cos(self.theta+action.r*time) - math.cos(self.theta))
                y = self.py - action.v / pow(action.r, 2) * (action.r * time * math.cos(self.theta + action.r * time) -
                                                             math.sin(self.theta+action.r*time) + math.sin(self.theta))
        return x, y


def check_destination(agents, min_interval=0.1):
    return [abs(agent.px - agent.gx) < min_interval and abs(agent.py - agent.gy) < min_interval
            for agent in agents]


class ENV(object):
    def __init__(self, agent_num, radius, v_pref):
        self.agent_num = agent_num
        self.agents = list()
        self.radius = radius
        self.v_pref = v_pref

    def get_joint_state(self, agent_idx):
        joint_states = JointState(*(self.agents[agent_idx].get_full_state() + self.agents[1-agent_idx].get_visible_state()))
        return joint_states

    def reset(self):
        # randomly initialize the agents' positions
        self.agents.append(Agent(55, 55, -75, -75, self.radius, self.v_pref, 0))
        self.agents.append(Agent(-55, 55, 75, -75, self.radius, self.v_pref, 0))

        return [self.get_joint_state(0), self.get_joint_state(1)]

    def compute_reward(self, agent_idx, action):
        agent = self.agents[agent_idx]
        other_agent = self.agents[1-agent_idx]
        # simple collision detection is done by checking the beginning and end position
        dmin = float('inf')
        for time in [0, 1]:
            self_pos = agent.compute_position(action, time, linear=False)
            other_pos = other_agent.compute_position(action, time, linear=True)
            distance = math.sqrt((self_pos[0]-other_pos[0])**2+(self_pos[1]-other_pos[1])**2)
            if distance < dmin:
                dmin = distance
        reaching_goal = abs(agent.px - agent.gx) < 0.1 and abs(agent.py - agent.gy) < 0.1
        if dmin < 0:
            reward = -0.25
        elif dmin < 0.2:
            reward = -0.1 - dmin/2
        elif reaching_goal:
            reward = 1
        else:
            reward = 0

        return reward

    def step(self, actions):
        """
        Take actions of all agents as input, output the rewards and states of each agent.
        :param actions:
        :return:
        """
        states = []
        rewards = []
        done_signals = []
        for agent_idx in range(self.agent_num):
            action = actions[agent_idx]
            reward = self.compute_reward(agent_idx, action)

            states.append(self.get_joint_state(agent_idx))
            rewards.append(reward)
            # TODO: stop if colliding with others?
            done_signals.append(reward == 1)

            # environment runs one step ahead
            self.agents[agent_idx].update_state(action)

        return states, rewards, done_signals
