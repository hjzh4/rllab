from rllab.algos import NOP
from rllab.baselines import ZeroBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs import normalize
from rllab.policies import UniformControlPolicy

env = normalize(CartpoleEnv())

policy = UniformControlPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
)

baseline = ZeroBaseline(env_spec=env.spec)

algo = NOP(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()
