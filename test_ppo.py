from spinup import ppo_tf1 as ppo
from spinup import vpg_tf1 as vpg
import tensorflow as tf
import gym
from spinup.utils.run_utils import ExperimentGrid
from gym.wrappers import FlattenObservation


def run_experiment(args):
    def env_fn():
        import offload_env  # registers custom envs to gym env registry
        env = gym.make('offload-v0')
        return env

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', env_fn)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 10000)
    eg.add('save_freq', 20)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
    eg.run(vpg, num_cpu=args.cpu)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--env_name', type=str, default="offload-v0")
    parser.add_argument('--exp_name', type=str, default='vpg-base')
    args = parser.parse_args()
    run_experiment(args)