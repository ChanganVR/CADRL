from collections import namedtuple
import math


FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])
JointState = namedtuple('JointState', ['px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta',
                                       'px1', 'py1', 'vx1', 'vy1', 'radius1'])
Velocity = namedtuple('Velocity', ['x', 'y'])
# v is velocity, under kinematic constraints, r is rotation angle otherwise it's speed direction
Action = namedtuple('Action', ['v', 'r'])


class Agent(object):
    def __init__(self, px, py, pgx, pgy, radius, v_pref, theta, kinematic_constrained):
        self.px = px
        self.py = py
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.pgx = pgx
        self.pgy = pgy
        self.v_pref = v_pref
        self.theta = theta
        self.kinematic_constrained = kinematic_constrained

    def update_state(self, action):
        self.px, self.py = self.compute_position(time=1, action=action)
        if self.kinematic_constrained:
            pass
        else:
            self.vx = math.cos(action.r) * action.v
            self.vy = math.sin(action.r) * action.v
            self.theta = 0

    def get_full_state(self):
        return self.px, self.py, self.vx, self.vy, self.radius, self.pgx, self.pgy, self.v_pref, self.theta

    def get_observable_state(self):
        return self.px, self.py, self.vx, self.vy, self.radius

    def compute_position(self, time, action=None):
        if action is None:
            # assume the agent travels in linear speed
            x = self.px + time * self.vx
            y = self.py + time * self.vy
        else:
            if self.kinematic_constrained:
                if action.r == 0:
                    x = self.px + time * action.v * math.cos(self.theta)
                    y = self.py + time * action.v * math.sin(self.theta)
                else:
                    x = self.px + action.v / pow(action.r, 2) * (time * math.sin(self.theta + action.r * time) +
                                                                 math.cos(self.theta+action.r*time) - math.cos(self.theta))
                    y = self.py - action.v / pow(action.r, 2) * (action.r * time * math.cos(self.theta + action.r * time) -
                                                                 math.sin(self.theta+action.r*time) + math.sin(self.theta))
            else:
                x = self.px + time * math.cos(action.r) * action.v
                y = self.py + time * math.sin(action.r) * action.v
        return x, y


class ENV(object):
    def __init__(self, config):
        self.radius = config.getfloat('agent', 'radius')
        self.v_pref = config.getfloat('agent', 'v_pref')
        self.kinematic_constrained = config.getboolean('agent', 'kinematic_constrained')
        self.agent_num = config.getint('sim', 'agent_num')
        self.xmin = config.getfloat('sim', 'xmin')
        self.xmax = config.getfloat('sim', 'xmax')
        self.ymin = config.getfloat('sim', 'ymin')
        self.ymax = config.getfloat('sim', 'ymax')
        self.crossing_radius = config.getfloat('sim', 'crossing_radius')
        self.agents = list()

    def get_joint_state(self, agent_idx):
        joint_states = JointState(*(self.agents[agent_idx].get_full_state() +
                                    self.agents[1-agent_idx].get_observable_state()))
        return joint_states

    def reset(self):
        # randomly initialize the agents' positions
        cr = self.crossing_radius
        self.agents.append(Agent(-cr, 0, cr, 0, self.radius, self.v_pref, 0, self.kinematic_constrained))
        self.agents.append(Agent(cr, 0, -cr, 0, self.radius, self.v_pref, 0, self.kinematic_constrained))

        return [self.get_joint_state(0), self.get_joint_state(1)]

    def compute_reward(self, agent_idx, action):
        agent = self.agents[agent_idx]
        other_agent = self.agents[1-agent_idx]
        # simple collision detection is done by checking the beginning and end position
        dmin = float('inf')
        for time in [0, 1]:
            self_pos = agent.compute_position(time, action)
            other_pos = other_agent.compute_position(time)
            distance = math.sqrt((self_pos[0]-other_pos[0])**2+(self_pos[1]-other_pos[1])**2)
            if distance < dmin:
                dmin = distance
        reached_goal = math.sqrt((agent.px - agent.pgx)**2 + (agent.py - agent.pgy)**2) < self.radius
        if dmin < 0:
            reward = -0.25
        elif dmin < 0.2:
            reward = -0.1 - dmin/2
        elif reached_goal:
            reward = 1
        else:
            reward = 0

        return reward

    def check_boundary(self, agent_idx):
        agent = self.agents[agent_idx]
        return self.xmin < agent.px < self.xmax and self.ymin < agent.py < self.ymax

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
            done_signals.append(reward == 1 or not self.check_boundary(agent_idx))

            # environment runs one step ahead
            self.agents[agent_idx].update_state(action)

        return states, rewards, done_signals
