# -*- coding: utf-8 -*-
import signal
import csv
import random
import math
import os
import time
import functools
import sys

import tensorflow as tf
import threading
import numpy as np
from pathlib import Path


from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier
from config_utils import load_config


import yaml



def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


def train_function(parallel_index):
    global global_t

    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > experiment_config['MAX_TIME_STEP']:
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op, score_input)
        global_t += diff_global_t

def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True


if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = None

experiment_config = load_config(config_path)
print(experiment_config)

device = "/cpu:0"
if experiment_config['USE_GPU']:
    device = "/gpu:0"





initial_learning_rate = log_uniform(experiment_config['INITIAL_ALPHA_LOW'],
                                    experiment_config['INITIAL_ALPHA_HIGH'],
                                    experiment_config['INITIAL_ALPHA_LOG_RATE'])

global_t = 0

stop_requested = False

if experiment_config['USE_LSTM']:
    global_network = GameACLSTMNetwork(experiment_config['ACTION_SIZE'], -1, device)
else:
    global_network = GameACFFNetwork(experiment_config['ACTION_SIZE'], -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                              decay=experiment_config['RMSP_ALPHA'],
                              momentum=0.0,
                              epsilon=experiment_config['RMSP_EPSILON'],
                              clip_norm=experiment_config['GRAD_NORM_CLIP'],
                              device=device)
fname = 'a3c_training_log_{}.csv'.format(experiment_config['GAME'])
a3c_log = Path(fname)
pis = ['PI_{}'.format(i) for i in range(experiment_config['ACTION_SIZE'])]
fieldnames = pis + ['V','Action','Score','Time',
                    'Step','Episode_reward','GameName']

if a3c_log.is_file():
    csvfile = open(fname,'a')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
else:
    csvfile = open(fname, 'w')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


for i in range(experiment_config['PARALLEL_SIZE']):
    training_thread = A3CTrainingThread(i, global_network,
                                        initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier,
                                        experiment_config['MAX_TIME_STEP'],
                                        device=device,
                                        config=experiment_config,
                                        logger=(csvfile,writer))

    training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)


summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(experiment_config['LOG_FILE'], sess.graph)


# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(experiment_config['CHECKPOINT_DIR'])
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print(">>> global step set: ", global_t)
    # set wall time
    wall_t_fname = experiment_config['CHECKPOINT_DIR'] + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
        wall_t = float(f.read())
else:
    print("Could not find old checkpoint")
    # set wall time
    wall_t = 0.0

train_threads = []

train_func_partial = functools.partial(train_function,
                                       wall_t=wall_t,
                                       sess=sess,
                                       summary_writer=summary_writer,
                                       summary_op=summary_op,
                                       score_input=score_input)


for i in range(experiment_config['PARALLEL_SIZE']):
    train_threads.append(
        threading.Thread(target=train_function, args=(i,)))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
    t.start()
    print('Hey')

print('Press Ctrl+C to stop')

# create_occasional_backup()


signal.pause()
print
"Nay"

print('Now saving data. Please wait')

for t in train_threads:
    t.join()

if not os.path.exists(experiment_config['CHECKPOINT_DIR']):
    os.mkdir(experiment_config['CHECKPOINT_DIR'])

    # write wall time
wall_t = time.time() - start_time
wall_t_fname = experiment_config['CHECKPOINT_DIR'] + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

saver.save(sess, experiment_config['CHECKPOINT_DIR'] + '/' + 'checkpoint', global_step=global_t)
