# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from datetime import datetime
import time

from game_state_gym import GameStateGym
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 config,
                 logger):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.config = config
        self.thread_log = logger[1]
        self.thread_logfile = logger[0]
        if config['USE_LSTM']:
            self.local_network = GameACLSTMNetwork(config['ACTION_SIZE'],
                                                   thread_index,
                                                   device)
        else:
            self.local_network = GameACFFNetwork(config['ACTION_SIZE'],
                                                 thread_index,
                                                 device)

        self.local_network.prepare_loss(config['ENTROPY_BETA'])

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameStateGym(113 * thread_index, config=config)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        step_dif = self.max_global_time_step - global_time_step

        learning_rate = self.initial_learning_rate * step_dif
        learning_rate /= self.max_global_time_step

        if learning_rate < 0.0:
            learning_rate = 0.0

        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op,
                      score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        if self.config['USE_LSTM']:
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(self.config['LOCAL_T_MAX']):
            game_state = self.game_state.s_t
            pi_, value_ = self.local_network.run_policy_and_value(sess,
                                                                  game_state)
            action = self.choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal
            self.episode_reward += reward
            if (self.thread_index == 0 and
                self.local_t % self.config['LOG_INTERVAL'] == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))



            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, global_t)

                self.episode_reward = 0
                self.game_state.reset()
                if self.config['USE_LSTM']:
                    self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + self.config['GAMMA'] * R
            td = R - Vi
            a = np.zeros([self.config['ACTION_SIZE']])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        if self.config['USE_LSTM']:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            sess.run(self.apply_gradients,
                     feed_dict={
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.local_network.initial_lstm_state: start_lstm_state,
                         self.local_network.step_size: [len(batch_a)],
                         self.learning_rate_input: cur_learning_rate})
        else:
            sess.run(self.apply_gradients,
                     feed_dict={
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.config['PERFORMANCE_LOG_INTERVAL']):
            self.prev_local_t += self.config['PERFORMANCE_LOG_INTERVAL']
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
            print('Current learning rate', cur_learning_rate)

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

