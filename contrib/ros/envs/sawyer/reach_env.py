"""
Reach task for the sawyer robot
"""

import numpy as np
from rllab.core.serializable import Serializable

from contrib.ros.envs.ros_env import RosEnv
from contrib.ros.robots.sawyer import Sawyer


class ReachEnv(RosEnv, Serializable):
    def __init__(self,
                 initial_robot_joint_pos,
                 initial_goal,
                 task_obj_mgr,
                 sparse_reward=False,
                 simulated=False):
        Serializable.quick_init(self, locals())

        self._distance_threshold = 0.05
        self._target_range = 0.15
        self._sparse_reward = sparse_reward
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()

        sawyer = Sawyer(
            initial_joint_pos=initial_robot_joint_pos, control_mode='position')
        RosEnv.__init__(
            self, task_obj_mgr=task_obj_mgr, robot=sawyer, simulated=simulated)

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()
        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=3)
        goal[:3] += random_goal_delta
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

        achieved_goal = np.array([
            self._robot.gripper_pose['position'].x,
            self._robot.gripper_pose['position'].y,
            self._robot.gripper_pose['position'].z
        ])

        return {
            'observation': robot_obs,
            'achieved_goal': achieved_goal,
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

    def _goal_distance(self, achieved_goal, goal):
        """
        :param achieved_goal:
        :param goal:
        :return distance: distance between goal_a and goal_b
        """
        assert achieved_goal.shape == goal.shape
        return np.linalg.norm(achieved_goal - goal, axis=-1)

    def done(self, achieved_goal, goal):
        """
        :return if_done: bool
                    if current episode is done:
        """
        return self._goal_distance(achieved_goal,
                                   goal) < self._distance_threshold
