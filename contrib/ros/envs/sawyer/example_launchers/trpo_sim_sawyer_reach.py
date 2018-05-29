import os.path as osp

from geometry_msgs.msg import Point
import numpy as np
import rospy

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import contrib.ros.envs.sawyer
from contrib.ros.envs.sawyer.reach_env import ReachEnv
from contrib.ros.envs.task_object_manager import TaskObject, TaskObjectManager

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.041662954890248294,
    'right_j1': -1.0258291091425074,
    'right_j2': 0.0293680414401436,
    'right_j3': 2.17518162913313,
    'right_j4': -0.06703022873354225,
    'right_j5': 0.3968371433926965,
    'right_j6': 1.7659649178699421,
}


def run_task(*_):
    model_dir = osp.join(
        osp.dirname(contrib.ros.envs.sawyer.__file__), 'models')
    initial_goal = np.array([0.6, -0.1, 0.8])
    target = TaskObject(
        name='target',
        initial_pos=Point(
            x=initial_goal[0], y=initial_goal[1], z=initial_goal[2]),
        random_delta_range=0.15,
        resource=osp.join(model_dir, 'target/model.sdf'))

    task_obj_mgr = TaskObjectManager()
    task_obj_mgr.add_target(target)

    rospy.init_node('trpo_sim_sawyer_reach_exp', anonymous=True)

    reach_env = ReachEnv(
        INITIAL_ROBOT_JOINT_POS, initial_goal, task_obj_mgr, simulated=True)

    rospy.on_shutdown(reach_env.shutdown)

    reach_env.initialize()

    env = TfEnv(normalize(reach_env))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        plot=False,
        force_batch_sampler=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    plot=False,
)
