import numpy as np
import rospy

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from contrib.ros.envs.sawyer.reach_env import ReachEnv
from contrib.ros.envs.task_object_manager import TaskObjectManager


def run_task(*_):
    initial_goal = np.array([0.6, -0.1, 0.8])

    task_obj_mgr = TaskObjectManager()

    rospy.init_node('trpo_real_sawyer_reach_exp', anonymous=True)

    reach_env = ReachEnv(initial_goal, task_obj_mgr, simulated=False)

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
