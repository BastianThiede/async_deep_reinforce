# -*- coding: utf-8 -*-
import sys

import cv2
from skimage import color
import gym
import numpy as np

## 0 no op
## 2 both
## 3 right
## 4 left
## 5 spring
## 6 go
#




class GameStateGym(object):
    def __init__(self, rand_seed, display=False, no_op_max=7,config=None):


        self.config = config
        self.env = gym.make(self.config['GAME'])
        self.env.reset()

        self._no_op_max = no_op_max


        self.display = display

        self.real_actions = self.config['ACTION_MAPPING']
        self.n_actions = len(self.real_actions)

        # height=210, width=160
        self._screen = np.empty((210, 160, 1), dtype=np.uint8)

        self.reset()

    def _process_frame(self, action, reshape):
        if self.display:
            self.env.render()
            #print "Current Action is: {}".format(action)
        screen,reward,terminal,meta = self.env.step(action)

        # screen shape is (210, 160)
        grey_screen = color.rgb2gray(screen)

        # resize to height=110, width=84
        resized_screen = cv2.resize(grey_screen, (84, 110))

        x_t = resized_screen[18:102, :]
        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))

        return reward, terminal, x_t

    def reset(self):
        self.env.reset()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def process(self, action):
        # convert original 18 action index to minimal action set index
        real_action = self.real_actions[action]

        r, t, x_t1 = self._process_frame(real_action, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)

    def update(self):
        self.s_t = self.s_t1
