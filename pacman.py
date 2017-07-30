''' Trains a naive agent using backpropagation with OpenAI Gym.'''
import numpy as np
import cPickle as pickle
import gym
import matplotlib.pyplot as plt
import argparse
import signal

# Configuration Parameters
input_image_size = (80,86)
size_layer1 = 200
size_output = 4
learning_rate = 0.0001

output_filename = 'weights.p'

''' Compute softmax values for each sets of scores in x. From Udacity Course '''
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

''' Convert the game image to neural net input '''
def img_to_input(image):
  # Input is 210x160x3 uint8.
  # Crop the screen
  image = np.reshape(image[0:171], (171, 160, 3))
  # Only take one color channel (green)
  image = image[:,:,1]
  # Threshold values to segment character, background, walls, and ghosts
  image[image == 111] = 0
  image[image == 28] = 1
  image[image > 1] = 2
  # Resample every 2 pixels to reduce input size
  image = image[::2, ::2]
  # Output as 1D 6880 float vector (80x86)
  return image.astype(np.float).ravel()

''' Run the input values through the network to get the output '''
def policy_forward(net_input):
  layer1_vals = np.dot(model['W1'], net_input)
  output_vals = np.dot(model['W2'], layer1_vals)
  probs = softmax(output_vals)
  return probs, layer1_vals # Return the probability of taking each action

''' Run the network in reverse to show how to update the weights
    action_rewards: shape(4,)'''
def policy_backward(net_input, action_rewards):
  rewards_layer1 = np.dot(model['W2'].T, action_rewards)
  dW2 = np.outer(action_rewards, rewards_layer1)
  dW1 = np.outer(rewards_layer1, net_input)
  return (dW1, dW2)

''' Pick an action probabilistically '''
def pick_action(action_probs):
  rand_val = np.random.uniform()
  rand_sum = action_probs[0]
  action = 1
  while rand_val > rand_sum:
    rand_sum += action_probs[action]
    action += 1
  return action

def save_weights():
  print("Saving weights to: "+output_filename)
  pickle.dump(model, open(output_filename, 'wb'))

def handle_sigint(signal, frame):
  save_weights()
  quit()

if __name__ == '__main__':

  # Setup argument parsing for settings and configuration
  ap = argparse.ArgumentParser()
  ap.add_argument('-l', '--load', required=False, help='Network weights to load from file')
  args = vars(ap.parse_args())

  # Load saved weights from file or initialize weights using Xavier initialization
  if (args['load'] != None):
    model = pickle.load(open(output_filename, 'rb'))
  else:
    # Initialize the weights using Xavier initialization
    model = {}
    model['W1'] = np.random.randn(size_layer1, input_image_size[0]*input_image_size[1]) / np.sqrt(input_image_size[0]*input_image_size[1])
    model['W2'] = np.random.randn(size_output, size_layer1) / np.sqrt(size_layer1)

  # Save the weights on sigint
  signal.signal(signal.SIGINT, handle_sigint)


  env = gym.make("MsPacman-v0")

  episode_number = 0
  reward_sum = 0
  next_action = 0

  observation = env.reset()

  # Run the network and updating
  while True:
    #env.render()

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
    
    # Calculate the reward for the action probabilites
    action_rewards = net_output*reward

    # Update the network weights each frame
    weight_updates = policy_backward(net_input, action_rewards)
    model['W1'] += weight_updates[0]*learning_rate
    model['W2'] += weight_updates[1]*learning_rate

    # Update rewards, frames, etc.
    reward_sum += reward
