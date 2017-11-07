# -*- coding: utf-8 -*-
import sys

import tensorflow as tf
import numpy as np

from game_state_gym import GameStateGym
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from rmsprop_applier import RMSPropApplier
from config_utils import load_config


if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = None

config = load_config(config_path)

def choose_action(pi_values):
  return np.random.choice(range(len(pi_values)), p=pi_values)

# use CPU for display tool
device = "/cpu:0"

if config['USE_LSTM']:
  global_network = GameACLSTMNetwork(config['ACTION_SIZE'], -1, device)
else:
  global_network = GameACFFNetwork(config['ACTION_SIZE'], -1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = config['RMSP_ALPHA'],
                              momentum = 0.0,
                              epsilon = config['RMSP_EPSILON'],
                              clip_norm = config['GRAD_NORM_CLIP'],
                              device = device)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

def load_newest_checkpoint():
    checkpoint = tf.train.get_checkpoint_state(config['CHECKPOINT_DIR'])
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
      print("Could not find old checkpoint")

load_newest_checkpoint()
game_state = GameStateGym(0, display=True, no_op_max=0,config=config)


while True:
  pi_values = global_network.run_policy(sess, game_state.s_t)
  print(pi_values)
  action = choose_action(pi_values)
  game_state.process(action)

  if game_state.terminal:
    game_state.reset()
  else:
    game_state.update()
