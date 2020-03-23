import gym
import offload_env
import random

NUM_EPISODES = 500

env = gym.make('offload-v0')

totalReturn_total_episodes = 0.0

action_taken_dict = {}
action_taken_dict['past_local'] = 0
action_taken_dict['past_cloud'] = 0
action_taken_dict['query_local'] = 0
action_taken_dict['query_cloud'] = 0

classification_error_total = 0.0
query_cost_total = 0.0

for i in range(NUM_EPISODES):
  observation = env.reset()
  done = False
  totalReturn_episode = 0.0
  info = {}
  while (not done):
    # Insert baseline policy for 'action':
    # a) local-model-only: '2'
    # b) cloud-model-only: '3'
    # c) random policy: 'random.randrange(0, 4)'
    action = random.randrange(0, 4)
    observation, reward, done, info = env.step(action)
    totalReturn_episode += reward

  total_class_error = info['classification_error_total']
  total_query_cost = info['query_cost_total']
  print("Epsiode {}      EpRet  {}     TotalClassificationError  {}    TotalQueryCost  {}".format(i, totalReturn_episode, total_class_error, total_query_cost))
  
  totalReturn_total_episodes += totalReturn_episode

  classification_error_total += info['classification_error_total']
  query_cost_total += info['query_cost_total']

  action_taken_dict['past_local'] += info['action_taken_dict']['past_local']
  action_taken_dict['past_cloud'] += info['action_taken_dict']['past_cloud']
  action_taken_dict['query_local'] += info['action_taken_dict']['query_local']
  action_taken_dict['query_cloud'] += info['action_taken_dict']['query_cloud']


print("------------------------")
AverageEpRet = totalReturn_total_episodes / float(NUM_EPISODES)
print("AverageEpRet: {}".format(AverageEpRet))
averageClassificationError = classification_error_total / float(NUM_EPISODES * 79)
print("AverageClassificationError: {}".format(averageClassificationError))
averageQueryCost = query_cost_total / float(NUM_EPISODES * 79)
print("AverageQueryCost: {}".format(averageQueryCost))
print('------------------------')

