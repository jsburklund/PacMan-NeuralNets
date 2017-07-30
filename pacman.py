''' Trains a naive agent using backpropagation with OpenAI Gym.'''

import numpy as np
import cPickle as pickle
import gym
import matplotlib.pyplot as plt
import argparse

# Configuration Parameters
input_image_size = (80,85)
size_layer1 = 200
size_output = 4


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

''' Compute softmax values for each sets of scores in x. From Udacity Course '''
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':

  # Setup argument parsing for settings and configuration
  ap = argparse.ArgumentParser()
  ap.add_argument('-l', '--load', required=False, help='Network weights to load from file')
  args = vars(ap.parse_args())

  # Optionally load saved weights from file
  # TODO

  # Initialize the weights using Xavier initialization
  model = {}

  env = gym.make("MsPacman-v0")

  episode_number = 0
  frame_num = 0
  reward_sum = 0
  next_action = 0

  plt.figure()
  observation = env.reset()

  # Run the network and updating
  while True:
    env.render()
    
    # Step the environment and get new measurements
    observation, reward, done, info = env.step(next_action)
    reward_sum += reward

    print(next_action)

    # Reset the environement if required
    if done:
      episode_number += 1
      env.reset()


    # Select the best action to take next
    next_action = 0
    
