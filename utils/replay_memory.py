from random import sample
from collections import namedtuple

import numpy as np

class ReplayBuffer(object):

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.insert_idx = 0

        assert self.capacity > 0

    @property
    def size(self):
        return len(self.buffer)

    def insert(self, state, action, reward, next_state, done):
        buffer_item = (state.copy(), action, reward, next_state.copy(), done)

        if self.size < self.capacity:
            self.buffer.append(buffer_item)
        else:
            self.buffer[self.insert_idx] = buffer_item
            self.insert_idx = (self.insert_idx + 1) % self.capacity

    def sample(self, batch_size):
        buffer_items = sample(self.buffer, batch_size)

        states = np.array([item[0] for item in buffer_items])
        actions = np.array([item[1] for item in buffer_items])
        rewards = np.array([item[2] for item in buffer_items])
        next_states = np.array([item[3] for item in buffer_items])
        dones = np.array([item[4] for item in buffer_items])

        return states, actions, rewards, next_states, dones


    