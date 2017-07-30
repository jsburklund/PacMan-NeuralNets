''' Continuously play episodes of the game to show the results of the weights'''
import numpy as np
import cPickle as pickle
import gym
import argparse
import pacman
from pacman import *

if __name__ == '__main__':

  # Setup argument parsing for settings and configuration
  ap = argparse.ArgumentParser()
  ap.add_argument('-w', '--weights', required=True, help='Network weights to load from file')
  args = vars(ap.parse_args())

  # Load saved weights from file or initialize weights using Xavier initialization
  pacman.model = pickle.load(open(output_filename, 'rb'))

  env = gym.make("MsPacman-v0")

  episode_number = 0
  reward_sum = 0
  next_action = 0

  observation = env.reset()

  # Run the network and updating
  while True:
    env.render()

    # Step the environment and get new measurements
    observation, reward, done, info = env.step(next_action)
    reward_sum += reward

    # Reset the environement if required
    if done:
      print('Episode '+str(episode_number)+': '+str(reward_sum))
      episode_number += 1
      reward_sum = 0
      env.reset()

    # Compute the results of the net
    net_input = img_to_input(observation)
    net_output, layer1_vals = policy_forward(net_input)

    # Select the best action to take next
    next_action = pick_action(net_output)
    
    # Update rewards, frames, etc.
    reward_sum += reward
