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
from contrib.ros.envs.sawyer.push_env import PushEnv
from contrib.ros.envs.task_object_manager import TaskObject, TaskObjectManager


def run_task(*_):
    model_dir = osp.join(osp.dirname(contrib.ros.envs.sawyer.__file__), 'models')

    table = TaskObject(name='table',
                       initial_pos=Point(x=0.75, y=0.0, z=0.0),
                       random_delta_range=0.15,
                       resource=osp.join(model_dir, 'cafe_table/model.sdf'))
    block = TaskObject(name='block',
                       initial_pos=Point(x=0.5725, y=0.1265, z=0.90),
                       random_delta_range=0.15,
                       resource=osp.join(model_dir, 'block/model.urdf'))
    initial_goal = np.array([0.6, -0.1, 0.80])
    target = TaskObject(name='target',
                        initial_pos=Point(x=initial_goal[0],
                                          y=initial_goal[1],
                                          z=initial_goal[2]),
                        random_delta_range=0.15,
                        resource=osp.join(model_dir, 'target/model.sdf'))

    task_obj_mgr = TaskObjectManager()
    task_obj_mgr.add_target(target)
    task_obj_mgr.add_common(table)
    task_obj_mgr.add_manipulated(block)

    rospy.init_node('trpo_sim_sawyer_push_exp', anonymous=True)

    push_env = PushEnv(initial_goal,
                       task_obj_mgr,
                       simulated=True)

    rospy.on_shutdown(push_env.shutdown)

    push_env.initialize()

    env = TfEnv(normalize(push_env))

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