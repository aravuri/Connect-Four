import net
import graphics
from graphics import Board
from net import DQN
from net import ReplayMemory
from net import Transition
from agent import *
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

policy_netP1 = DQN()
policy_netP1.load_state_dict(torch.load('p1.pth'))
policy_netP2 = DQN()
policy_netP2.load_state_dict(torch.load('p2.pth'))
policy_netP1.eval()
policy_netP2.eval()

b = Board()
b.play(BotAgent(policy_netP1), HumanAgent())
