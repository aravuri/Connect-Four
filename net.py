import random
import math
from collections import namedtuple, deque
from graphics import Board
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

n_actions = 7

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv_2d1 = nn.Conv2d(1, 16, kernel_size=4, padding='same')
        self.conv_2d2 = nn.Conv2d(16, 32, kernel_size=4, padding='same')
        self.conv_2d3 = nn.Conv2d(32, 32, kernel_size=4, padding='same')
        self.conv_2d4 = nn.Conv2d(32, 32, kernel_size=4, padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*6, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.conv_2d1(x))
        x = F.leaky_relu(self.conv_2d2(x))
        x = F.leaky_relu(self.conv_2d3(x))
        x = F.leaky_relu(self.conv_2d4(x))
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)