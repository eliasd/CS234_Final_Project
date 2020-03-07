import gym
import offload_env

env = gym.make('offload-v0')

observation = env.reset()
done = False
totalReward_pastLocal = 0.0
while(not done):
  action = 0
  observation, reward, done, info = env.step(action)
  totalReward_pastLocal += reward

observation = env.reset()
done = False
totalReward_pastCloud = 0.0
while(not done):
  action = 1
  observation, reward, done, info = env.step(action)
  totalReward_pastCloud += reward

observation = env.reset()
done = False
totalReward_queryLocal = 0.0
while(not done):
  action = 2
  observation, reward, done, info = env.step(action)
  totalReward_queryLocal += reward

observation = env.reset()
done = False
totalReward_queryCloud = 0.0
while(not done):
  action = 3
  observation, reward, done, info = env.step(action)
  totalReward_queryCloud += reward

import random

observation = env.reset()
done = False
totalReward_random = 0.0
while(not done):
  action = random.randrange(0, 4)
  observation, reward, done, info = env.step(action)
  totalReward_random += reward

print("ONLY USE PAST LOCAL: {}".format(totalReward_pastLocal))
print("ONLY USE PAST CLOUD: {}".format(totalReward_pastCloud))
print("ONLY QUERY LOCAL: {}".format(totalReward_queryLocal))
print("ONLY QUERY CLOUD: {}".format(totalReward_queryCloud))
print("RANDOM: {}".format(totalReward_random))