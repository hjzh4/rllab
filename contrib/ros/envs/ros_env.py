from geometry_msgs.msg import Pose, Point
import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc.ext import get_seed
from rllab.spaces import Box

from contrib.ros.util.common import rate_limited
from contrib.ros.util.gazebo import Gazebo
from contrib.ros.util.vicon import Vicon


class RosEnv(Env, Serializable):
    """
    Superclass for all ros environment
    """

    def __init__(self, task_obj_mgr, robot, simulated=False):
        """
        :param simulated: bool
                if the environment is for real robot or simulation
        :param robot: object
                the robot interface for the environment
        :param task_obj_mgr: object
                Use this to manage objects used in a specific task
        """
        Serializable.quick_init(self, locals())

        np.random.RandomState(get_seed())

        self._robot = robot

        self.simulated = simulated

        self.task_obj_mgr = task_obj_mgr

        if self.simulated:
            self.gazebo = Gazebo()
            self._initial_setup()
            self.task_obj_mgr.sub_gazebo()
        else:
            self.vicon = Vicon()
            self._initial_setup()
            self.task_obj_mgr.sub_vicon()

        # Verify robot is enabled
        if not self._robot.enabled:
            raise RuntimeError('The robot is not enabled!')
            # TODO (gh/74: Add initialize interface for robot)

    def initialize(self):
        # TODO (gh/74: Add initialize interface for robot)
        pass

    def shutdown(self):
        if self.simulated:
            # delete model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.delete_gazebo_model(obj.name)
        else:
            ready = False
            while not ready:
                ans = input('Are you ready to exit?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

    # =======================================================
    # The functions that base rllab Env asks to implement
    # =======================================================
    @rate_limited(100)
    def step(self, action):
        """
        Perform a step in gazebo. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._robot.set_command(action)

        obs = self.get_obs()

        reward = self.reward(obs['achieved_goal'], self.goal)
        done = self.done(obs['achieved_goal'], self.goal)
        next_observation = obs['observation']
        return Step(observation=next_observation, reward=reward, done=done)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._robot.move_to_start_position()

        self.goal = self.sample_goal()

        if self.simulated:
            target_idx = 0
            for target in self.task_obj_mgr.targets:
                self.gazebo.set_model_pose(
                    model_name=target.name,
                    new_pose=Pose(
                        position=Point(
                            x=self.goal[target_idx * 3],
                            y=self.goal[target_idx * 3 + 1],
                            z=self.goal[target_idx * 3 + 2])))
            self._reset_sim()
        else:
            self._reset_real()
        obs = self.get_obs()
        initial_observation = obs['observation']

        return initial_observation

    def _reset_real(self):
        """
        reset the real
        """
        # randomize start position of manipulateds
        for manipulated in self.task_obj_mgr.manipulateds:
            manipulated_random_delta = np.zeros(2)
            new_pos = manipulated.initial_pos
            while np.linalg.norm(manipulated_random_delta) < 0.1:
                manipulated_random_delta = np.random.uniform(
                    -manipulated.random_delta_range,
                    manipulated.random_delta_range,
                    size=2)
            new_pos[0] += manipulated_random_delta[0]
            new_pos[1] += manipulated_random_delta[1]
            logger.log('new position for {} is x = {}, y = {}, z = {}'.format(
                manipulated.name, new_pos[0], new_pos[1], new_pos[2]))
            ready = False
            while not ready:
                ans = input(
                    'Have you finished setting up {}?[Yes/No]\n'.format(
                        manipulated.name))
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

    def _reset_sim(self):
        """
        reset the simulation
        """
        # randomize start position of manipulateds
        for manipulated in self.task_obj_mgr.manipulateds:
            manipulated_random_delta = np.zeros(2)
            while np.linalg.norm(manipulated_random_delta) < 0.1:
                manipulated_random_delta = np.random.uniform(
                    -manipulated.random_delta_range,
                    manipulated.random_delta_range,
                    size=2)
            self.gazebo.set_model_pose(
                manipulated.name,
                new_pose=Pose(
                    position=Point(
                        x=manipulated.initial_pos.x +
                        manipulated_random_delta[0],
                        y=manipulated.initial_pos.y +
                        manipulated_random_delta[1],
                        z=manipulated.initial_pos.z)))

    @property
    def action_space(self):
        return self._robot.action_space

    @property
    def observation_space(self):
        return Box(-np.inf, np.inf, shape=self.get_obs()['observation'].shape)

    def _initial_setup(self):
        self._robot.move_to_start_position()

        if self.simulated:
            # Generate the world
            # Load the model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.load_gazebo_model(obj)
        else:
            ready = False
            while not ready:
                ans = input('Have you finished your initial setup?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

    # ====================================================
    # Need to be implemented in specific task env
    # ====================================================
    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        raise NotImplementedError

    def get_obs(self):
        """
        Get observation
        """
        raise NotImplementedError

    def done(self, achieved_goal, goal):
        """
        :return if done: bool
        """
        raise NotImplementedError

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        """
        raise NotImplementedError

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value
