import net
import graphics
from graphics import Board
from net import DQN
from net import ReplayMemory
from net import Transition
from itertools import count
from agent import *
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_actions = 7
BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10
policy_netP1 = DQN()
policy_netP2 = DQN()
target_netP1 = DQN()
target_netP2 = DQN()
policy_netP1.load_state_dict(torch.load('p1.pth'))
target_netP1.load_state_dict(torch.load('p1.pth'))
policy_netP2.load_state_dict(torch.load('p2.pth'))
target_netP2.load_state_dict(torch.load('p2.pth'))
target_netP1.eval()
target_netP2.eval()

steps_done = 0

memoryP1 = ReplayMemory(10000)
memoryP2 = ReplayMemory(10000)
optimizerP1 = optim.RMSprop(policy_netP1.parameters())
optimizerP2 = optim.RMSprop(policy_netP2.parameters())


def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def select_best_action(state, policy_net):
    return policy_net(state).max(1)[1].view(1, 1)

episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train_with_P2():
    num_episodes = 25000
    for i_episode in range(num_episodes):
        b = Board()
        state = b.getBoard()
        state2 = None
        actionP1 = None
        actionP2 = None
        rewardP1 = 0
        rewardP2 = 0
        done = False
        for t in count():
            actionP1 = select_action(state, policy_netP1)
            works = b.put(1, actionP1)
            rewardP1 = 0
            if not works:
                done = True
                rewardP1 = -10
                rewardP2 = 0
                print("Episode " + str(i_episode) + " Winner: Out of Bounds P1, Time=" + str(t))
            elif b.getWinner() == 1:
                done = True
                rewardP1 = 1
                rewardP2 = -1
                print("Episode " + str(i_episode) + " Winner: P1, Time=" + str(t))
            if done:
                memoryP1.push(state, actionP1, None, torch.tensor([rewardP1]))
                optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
                memoryP2.push(state2, actionP2, None, torch.tensor([rewardP2]))
                optimize_model(memoryP2, policy_netP2, target_netP2, optimizerP2)
                break

            next_state2 = b.getBoard()
            if state2 is not None:
                memoryP2.push(state2, actionP2, next_state2, torch.tensor([rewardP2]))
                optimize_model(memoryP2, policy_netP2, target_netP2, optimizerP2)
            state2 = next_state2
            actionP2 = select_action(state2, policy_netP2)
            works = b.put(2, actionP2)
            rewardP2 = 0
            if not works:
                done = True
                rewardP1 = 0
                rewardP2 = -10
                print("Episode " + str(i_episode) + " Winner: Out of Bounds P2, Time=" + str(t))
            elif b.getWinner() == 2:
                done = True
                rewardP1 = -1
                rewardP2 = 1
                print("Episode " + str(i_episode) + " Winner: P2, Time=" + str(t))
            if done:
                memoryP1.push(state, actionP1, None, torch.tensor([rewardP1]))
                optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
                memoryP2.push(state2, actionP2, None, torch.tensor([rewardP2]))
                optimize_model(memoryP2, policy_netP2, target_netP2, optimizerP2)
                break
            next_state = b.getBoard()
            memoryP1.push(state, actionP1, next_state, torch.tensor([rewardP1]))
            optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
            state = next_state

        if i_episode % TARGET_UPDATE == 0:
            target_netP1.load_state_dict(policy_netP1.state_dict())
            target_netP2.load_state_dict(policy_netP2.state_dict())

def trainP1Only():
    num_episodes = 500

    for i_episode in range(num_episodes):
        b = Board()
        state = b.getBoard()
        action = None
        reward = 0
        done = False
        r = RandomAgent()
        for t in count():
            action = select_action(state, policy_netP1)
            works = b.put(1, action)
            reward = 0
            if not works:
                reward = -10
                done = True
                print("Episode " + str(i_episode) + " Winner: Out of Bounds")
            elif b.getWinner() == 1:
                reward = 1
                done = True
                print("Episode " + str(i_episode) + " Winner: P1")
            if done:
                memoryP1.push(state, action, None, torch.tensor([reward]))
                optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
                break

            r.make_move(b, 2)
            if b.getWinner() == 2:
                done = True
                reward = -1
                print("Episode " + str(i_episode) + " Winner: P2")
            if done:
                memoryP1.push(state, action, None, torch.tensor([reward]))
                optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
                break
            next_state = b.getBoard()
            memoryP1.push(state, action, next_state, torch.tensor([reward]))
            optimize_model(memoryP1, policy_netP1, target_netP1, optimizerP1)
            state = next_state

        if i_episode % TARGET_UPDATE == 0:
            target_netP1.load_state_dict(policy_netP1.state_dict())
            target_netP2.load_state_dict(policy_netP2.state_dict())


train_with_P2()
torch.save(policy_netP1.state_dict(), 'p1.pth')
torch.save(policy_netP2.state_dict(), 'p2.pth')
# b = Board()
# b.playBots(policy_netP1, policy_netP2, select_best_action)