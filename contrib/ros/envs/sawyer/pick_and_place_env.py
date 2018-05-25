"""
Pick-and-place task for the sawyer robot
"""

import numpy as np
from rllab.core.serializable import Serializable

from contrib.ros.envs.ros_env import RosEnv
from contrib.ros.robots.sawyer import Sawyer

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.041662954890248294,
    'right_j1': -1.0258291091425074,
    'right_j2': 0.0293680414401436,
    'right_j3': 2.17518162913313,
    'right_j4': -0.06703022873354225,
    'right_j5': 0.3968371433926965,
    'right_j6': 1.7659649178699421,
}


class PickAndPlaceEnv(RosEnv, Serializable):
    def __init__(self,
                 initial_goal,
                 task_obj_mgr,
                 sparse_reward=True,
                 simulated=False):
        Serializable.quick_init(self, locals())

        self._distance_threshold = 0.05
        self._target_range = 0.15
        self._sparse_reward = sparse_reward
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()

        sawyer = Sawyer(initial_joint_pos=INITIAL_ROBOT_JOINT_POS,
                        control_mode='position')

        RosEnv.__init__(
            self,
            task_obj_mgr=task_obj_mgr,
            robot=sawyer,
            simulated=True)

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()
        random_goal_delta = np.random.uniform(-self._target_range,
                                              self._target_range,
                                              size=2)
        goal[:2] += random_goal_delta
        return goal

    def get_obs(self):
        """
        Get Observation
        :return observation: dict
                    {'observation': obs,
                     'achieved_goal': achieved_goal,
                     'desired_goal': self.goal}
        """
        robot_obs = self._robot.get_obs()

        manipulated_obs = self.task_obj_mgr.get_manipulateds_obs()

        obs = np.concatenate((robot_obs, manipulated_obs['obs']))

        return {
            'observation': obs,
            'achieved_goal': manipulated_obs['achieved_goal'],
            'desired_goal': self.goal
        }

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        :param achieved_goal: the current gripper's position or object's position in the current training episode.
        :param goal: the goal of the current training episode, which mostly is the target position of the object or the
                     position.
        :return reward: float
                    if sparse_reward, the reward is -1, else the reward is minus distance from achieved_goal to
                    our goal. And we have completion bonus for two kinds of types.
        """
        d = self._goal_distance(achieved_goal, goal)
        if d < self._distance_threshold:
            return 88
        else:
            if self._sparse_reward:
                return -1.
            else:
                return -d

    def _goal_distance(self, goal_a, goal_b):
        """
        :param goal_a:
        :param goal_b:
        :return distance: distance between goal_a and goal_b
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def done(self, achieved_goal, goal):
        """
        :return if_done: bool
                    if current episode is done:
        """
        return self._goal_distance(achieved_goal, goal) < self._distance_threshold
