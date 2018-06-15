import math
from utils import JointState, Action


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
        self.done = False

    def update_state(self, action, time):
        self.px, self.py = self.compute_position(time=time, action=action)
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
            # assume the agent travels in original speed
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
        self.max_time = config.getint('sim', 'max_time')
        self.agents = [None, None]
        self.counter = 0

    def compute_joint_state(self, agent_idx):
        if self.agents[agent_idx].done:
            return None
        else:
            return JointState(*(self.agents[agent_idx].get_full_state() +
                              self.agents[1-agent_idx].get_observable_state()))

    def reset(self):
        # randomly initialize the agents' positions
        cr = self.crossing_radius
        self.agents[0] = Agent(-cr, 0, cr, 0, self.radius, self.v_pref, 0, self.kinematic_constrained)
        self.agents[1] = Agent(cr, 0, -cr, 0, self.radius, self.v_pref, 0, self.kinematic_constrained)
        self.counter = 0

        return [self.compute_joint_state(0), self.compute_joint_state(1)]

    def compute_reward(self, agent_idx, actions):
        """
        When performing one-step lookahead, only one action is known, the position of the other agent is approximate
        When called by step(), both actions are known, the position of the other agent is exact
        """
        agent = self.agents[agent_idx]
        other_agent = self.agents[1-agent_idx]
        # simple collision detection is done by checking the beginning and end position
        dmin = float('inf')
        dmin_time = 1
        for time in [0, 0.5, 1]:
            pos = agent.compute_position(time, actions[agent_idx])
            other_pos = other_agent.compute_position(time, actions[1-agent_idx])
            distance = math.sqrt((pos[0]-other_pos[0])**2 + (pos[1]-other_pos[1])**2)
            if distance < dmin:
                dmin = distance
                dmin_time = time
        final_pos = agent.compute_position(1, actions[agent_idx])
        reached_goal = math.sqrt((final_pos[0] - agent.pgx)**2 + (final_pos[1] - agent.pgy)**2) < self.radius

        if dmin < self.radius * 2:
            reward = -0.25
            end_time = dmin_time
        else:
            end_time = 1
            if dmin < self.radius * 2 + 0.2:
                reward = -0.1 - dmin/2
            elif reached_goal:
                reward = 1
            else:
                reward = 0

        return reward, end_time

    def check_boundary(self, agent_idx):
        agent = self.agents[agent_idx]
        return self.xmin < agent.px < self.xmax and self.ymin < agent.py < self.ymax

    def step(self, actions):
        """
        Take actions of all agents as input, output the rewards and states of each agent.
        Hitting the boundary or exceeding the maximum time will emit the done signal, but not negative reward
        """
        rewards = []
        done_signals = []
        end_times = []
        for agent_idx in range(self.agent_num):
            reward, end_time = self.compute_reward(agent_idx, actions)
            rewards.append(reward)
            end_times.append(end_time)

        # collision is mutual
        if rewards[0] == -0.25 or rewards[1] == -0.25:
            assert rewards[0] == rewards[1]

        for agent_idx in range(2):
            self.agents[agent_idx].update_state(actions[agent_idx], end_times[agent_idx])
        states = [(self.compute_joint_state(agent_idx)) for agent_idx in range(2)]

        for agent_idx in range(2):
            agent = self.agents[agent_idx]
            reward = rewards[agent_idx]
            if not agent.done:
                # only update agent's status if it's active
                if reward == 1:
                    agent.done = 1
                elif reward == -0.25:
                    agent.done = 2
                elif not self.check_boundary(agent_idx):
                    agent.done = 3
                elif self.counter > self.max_time:
                    agent.done = 4
                else:
                    agent.done = False
            done_signals.append(agent.done)

        self.counter += 1

        return states, rewards, done_signals

