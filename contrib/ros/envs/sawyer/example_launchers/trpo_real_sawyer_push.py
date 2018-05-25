from geometry_msgs.msg import Point
import numpy as np
import rospy

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from contrib.ros.envs.sawyer.push_env import PushEnv
from contrib.ros.envs.task_object_manager import TaskObject, TaskObjectManager


def run_task(*_):
    vicon_iser_block = TaskObject(
        name='vicon_iser_block',
        initial_pos=Point(x=0.5725, y=0.1265, z=0.90),
        random_delta_range=0.15,
        resource='vicon/iser_block/iser_block')

    initial_goal = np.array([0.6, -0.1, 0.80])

    task_obj_mgr = TaskObjectManager()
    task_obj_mgr.add_manipulated(vicon_iser_block)

    rospy.init_node('trpo_real_sawyer_push_exp', anonymous=True)

    push_env = PushEnv(initial_goal, task_obj_mgr, simulated=False)

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
