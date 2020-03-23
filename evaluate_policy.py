from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
import offload_env

# Replace 'policy_path' with file path to the 
# output policy of the train_*.py files:
policy_path = 'VPG_RUNS/vpg-base/vpg-base_s40'
_, get_action =load_policy_and_env(policy_path)

env = gym.make('offload-v0')
run_policy(env, get_action)
