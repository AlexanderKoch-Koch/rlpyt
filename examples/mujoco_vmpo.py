
"""
Runs one instance of the Mujoco environment and optimizes using V-MPO algorithm.
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.algos.pg.v_mpo import VMPO
from rlpyt.agents.pg.gaussian_vmpo_agent import MujocoVmpoAgent
from rlpyt.models.pg.mujoco_ff_model import MujocoVmpoFfModel
from rlpyt.envs.gym import make as gym_make
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.launching.affinity import make_affinity


def build_and_train(id="Ant-v3", run_ID=0, cuda_idx=None):
    affinity = make_affinity(n_cpu_core=24, cpu_per_run=24, n_gpu=0, set_affinity=True)
    sampler = CpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=id),
        eval_env_kwargs=dict(id=id),
        batch_T=40,  # Four time-steps per sampler iteration.
        batch_B=64 * 100,
        max_decorrelation_steps=100,
        eval_n_envs=1,
        eval_max_steps=int(10e8),
        eval_max_trajectories=8,
    )
    algo = VMPO(T_target_steps=100, pop_art_reward_normalization=True, discrete_actions=False, epochs=1)
    agent = MujocoVmpoAgent(ModelCls=MujocoVmpoFfModel, model_kwargs=dict(linear_value_output=False, layer_norm=True))
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(1e10),
        log_interval_steps=int(1e6),
        affinity=affinity
    )
    config = dict(id=id)
    name = "vmpo_" + id
    log_dir = "vmpo_mujoco"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', help='gym env id', default='Ant-v3')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        id=args.id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
